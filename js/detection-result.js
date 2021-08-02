/** Abstraction of a face detection result. */
class DetectionResult {
    /**
     * Creates an instance of a detection result.
     */
    constructor() {
        this.box = cv.Rect();
        this.distance = 0.0;
        // this.resultHeatMap = new cv.Mat();
        this.resultRaw = new cv.Mat();
        this.scale = 0.0;
        this.score = 0.0;
        this.templateRgb = new cv.Mat();
        this.templateReference = new TemplateType();
    }

    /**
     * Disposes this instance.
     */
    dispose() {
        // don't dispose TemplateReference
        // this.resultHeatMap;
        this.resultRaw.delete();
        this.templateRgb.delete();
        this.templateReference.dispose();
    }
}