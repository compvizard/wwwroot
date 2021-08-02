/** Collection of face templates. */
class FaceTemplates {
    /**
     * Creates an instance of face templates.
     */
    constructor() {
        this.maxSize = new cv.Size(0, 0);
        this.reference = new Array();
        this.templates = new Array();
    }

    /**
     * Initializes this instance.
     * @param {cv.Mat} faceImg The face image.
     * @param {cv.Size} maxSize The maximum size.
     */
    init(faceImg, maxSize) {
        this.maxSize = maxSize;
        this.dispose();
        this.reference = this.generateTemplates(faceImg, 0.1, 1, 0.1, maxSize);
        this.templates = this.generateTemplates(faceImg, 0.6, 1.1, 0.05, maxSize);
    }

    /**
     * Disposes this instance.
     */
    dispose() {
        this.reference.forEach(e => {
            e.dispose();
        });
        this.templates.forEach(e => {
            e.dispose();
        });
    }

    /**
     * Generates an array of augmented face templates.
     * @param {cv.Mat} faceImg The face image.
     * @param {number} smallest The smallest scale.
     * @param {number} largest The largest scale.
     * @param {number} step The scale augmentation step.
     * @param {cv.Size} maxSize The maximum size.
     * @returns {Array.<TemplateType>} Array of face templates.
     */
    generateTemplates(faceImg, smallest, largest, step, maxSize) {
        if (faceImg == null) {
            throw "faceImg cannot be null when generating templates";
        }

        var q = new Array();

        for (var s = largest; s >= smallest; s -= step) {
            // filter out over-scaling
            var w = Math.round(s * faceImg.width);
            var h = Math.round(s * faceImg.height);
            if (w >= maxSize.width || h >= maxSize.height) continue;

            // filter out under-scaling
            if (w < 20 || h < 20) break;

            let rescaled = new cv.Mat();
            cv.resize(faceImg, rescaled, new cv.Size(0, 0), s, s, cv.INTER_AREA);
            var bkg = cv.mean(rescaled)[0];
            var center = new cv.Point(rescaled.cols / 2, rescaled.rows / 2);
            for (var a = -15; a <= 15; a += 15) {
                let M = cv.getRotationMatrix2D(center, a, 1);
                var rotated = new cv.Mat();
                cv.warpAffine(rescaled, rotated, M, rescaled.size(), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(bkg));
                var t = new TemplateType();
                t.scale = s;
                t.template = rotated;
                t.isOriginal = s == 1;
                q.push(t);
            }
        }

        return q;
    }

    /**
     * Updates the template collection with new face image.
     * @param {cv.Mat} faceImg The face image.
     * @param {number} smallest The smallest scale.
     * @param {number} largest The largest scale.
     * @param {number} thresh The min. score for a good template.
     */
    update(faceImg, smallest, largest, thresh) {
        // validate face image first
        var maxScore = 0.0;
        var faceSize = faceImg.size();
        this.reference.forEach(r => {
            let refNorm = new cv.Mat();
            cv.resize(r.template, refNorm, { width: faceSize.width, height: faceSize.height }, cv.INTER_NEAREST);
            let res = new cv.Mat();
            let mask = new cv.Mat();
            cv.matchTemplate(faceImg, refNorm, res, cv.TM_CCOEFF_NORMED, mask);
            var score = res.ucharPtr(0, 0)[0];
            if (score > maxScore)
                maxScore = score;
            else
                refNorm.delete();
        });

        if (maxScore > thresh) {
            // update the templates
            this.templates.forEach(e => {
                e.dispose();
            });
                this.templates = this.generateTemplates(faceImg, smallest, largest, 0.05, this.maxSize);
        }
    }
}
