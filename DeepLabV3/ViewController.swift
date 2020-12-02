//
//  ViewController.swift
//  DeepLabV3
//
//  Created by Pearl Chye on 10/1/20.
//  Copyright © 2020 Isobar. All rights reserved.
//

import UIKit
import Vision

class ViewController: UIViewController {
    
    @IBOutlet weak var mainImageView: UIImageView!
    @IBOutlet weak var drawingView: DrawingSegmentationView!
    @IBOutlet weak var detectionView: UIView!
    @IBOutlet weak var activity: UIActivityIndicatorView!
    
    private var detectionOverlay: CALayer! = nil
    
    let segmentationModel = DeepLabV3()
    let objectModel = YOLOv3Tiny()
    var request: VNCoreMLRequest?
    var request2: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var visionModel2: VNCoreMLModel?
    
    let imageList = ["5","1","2","3","4"]
    var imgIndex = 0;
    
    override func viewDidLoad() {
        super.viewDidLoad()

        detectionOverlay = CALayer() // container layer that has all the renderings of the observations
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: detectionView.bounds.width,
                                         height: detectionView.bounds.height)
        detectionOverlay.setAffineTransform(CGAffineTransform(rotationAngle: 0).scaledBy(x: 1, y: -1))
        detectionOverlay.position = CGPoint(x: detectionView.bounds.midX, y: detectionView.bounds.midY)
        detectionView.layer.addSublayer(detectionOverlay)
        
        
        
        // setup ml model
        setUpModel()
        startProcess()
    }
    
    func startProcess(){
        DispatchQueue.main.async {
            self.activity.startAnimating()
            self.activity.isHidden = false
        }
        
        var image:CIImage = CIImage(image: mainImageView.image!)!
        image = image.transformed(by: CGAffineTransform(scaleX: 513.0/image.extent.width, y: 513.0/image.extent.height))
        let context = CIContext(options: nil)
        let cgimage = context.createCGImage(image, from: image.extent)
        self.predict(with: cgimage!)
    }
    
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: segmentationModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .centerCrop
        } else {
            fatalError()
        }
        
        if let visionModel2 = try? VNCoreMLModel(for: objectModel.model) {
           self.visionModel2 = visionModel2
           request2 = VNCoreMLRequest(model: visionModel2, completionHandler: vision2RequestDidComplete)
       } else {
           fatalError()
       }
    }
    
    @IBAction func onTap(){
        imgIndex += 1
        if imgIndex == imageList.count {
            imgIndex = 0;
        }
        
        mainImageView.image = UIImage(named: imageList[imgIndex])
        startProcess()
    }

}

extension ViewController {
    // prediction
    func predict(with image: CGImage) {
        guard let request = request else { fatalError() }
        guard let request2 = request2 else { fatalError() }
        
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try? handler.perform([request])
        try? handler.perform([request2])
    }
    
    // post-processing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let segmentationmap = observations.first?.featureValue.multiArrayValue {
            
            drawingView.segmentationmap = SegmentationResultMLMultiArray(mlMultiArray: segmentationmap)
        }
    }
    
    func vision2RequestDidComplete(request: VNRequest, error: Error?) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionOverlay.sublayers = nil
        if let observations = request.results as? [VNRecognizedObjectObservation]{
            let imageSize = self.detectionView.bounds.size
            for observation in observations {
                // Select only the label with the highest confidence.
                let topLabelObservation = observation.labels[0]
                let objectBounds = VNImageRectForNormalizedRect(observation.boundingBox, Int(imageSize.width), Int(imageSize.height))
                
                let shapeLayer = self.createRoundedRectLayerWithBounds(objectBounds)
                
                let textLayer = self.createTextSubLayerInBounds(objectBounds,
                                                                identifier: topLabelObservation.identifier,
                                                                confidence: topLabelObservation.confidence)
                shapeLayer.addSublayer(textLayer)
                detectionOverlay.addSublayer(shapeLayer)
                print(topLabelObservation)
            }
        }
        CATransaction.commit()
        
        DispatchQueue.main.async {
            self.activity.stopAnimating()
        }
    }
    
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String, confidence: VNConfidence) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let formattedString = NSMutableAttributedString(string: String(format: "\(identifier)\nConfidence:  %.2f", confidence))
        let largeFont = UIFont(name: "Helvetica", size: 24.0)!
        formattedString.addAttributes([NSAttributedString.Key.font: largeFont], range: NSRange(location: 0, length: identifier.count))
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.width-20, height: bounds.size.height-20)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        textLayer.shadowOpacity = 0.7
        textLayer.shadowOffset = CGSize(width: 2, height: 2)
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
        textLayer.contentsScale = 2.0 // retina rendering
        // rotate the layer into screen orientation and scale and mirror
        textLayer.setAffineTransform(CGAffineTransform(scaleX: 1, y: -1))
        return textLayer
    }
    
    func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.borderWidth = 5
        shapeLayer.borderColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 0.2, 0.4])
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
    
}
