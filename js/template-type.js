/** Abstraction of a face template. */
class TemplateType {
    /**
     * Creates an instance of a face template.
     */
    constructor() {
        this.isOriginal = false;
        this.scale = 1.0;
        this.template = new cv.Mat();
    }

    /**
     * Disposes this instance.
     */
    dispose() {
        this.template.delete();
    }
}
