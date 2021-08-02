const detectionStage = {
    NONE: "none",
    DETECT_FACE: "detect_face",
    MATCH_FACE: "match_face"
}

const genderType = {
    MALE: "male",
    FEMALE: "female"
}

/** Abstraction of the face tracker. */
class FaceTracker {

    constructor() {
        this.stage = detectionStage.NONE;
        this.focalLen = 0;
        this.observedFaceHgt = 0;
        this.actualFaceHgt = 0;
        this.gender = genderType.MALE;
        this.userHeight = 175;
        this.faceTmpls = new FaceTemplates();
        this.grayImage = {};
        this.isInit = false;
        this.smallest = 0.6;
        this.largest = 1.5;
        this.matchThresh = 0.75;
        this.updateThresh = 0.75;
        this.tune = 0;
    }

    /**
     * Starts face tracking.
     */
    lock() {
        this.stage = detectionStage.DETECT_FACE;
    }

    /**
     * Stops face tracking.
     */
    unlock() {
        this.stage = detectionStage.NONE;
        this.focalLen = 0;
    }

    /**
     * Detects the face and computes the distance from the camera.
     * @param {cv.Mat} rgbImage The source image.
     * @returns {cv.Mat} The detection result.
     */
    detect(rgbImage) {
        if (rgbImage == null) {
            throw "rgbImage is null";
        }

        if (!this.isInit) {
            this.isInit = true;
            var width = rgbImage.cols;
            var height = rgbImage.rows;
            this.grayImage = new cv.Mat(height, width, cv.CV_8UC1);
        }
        cv.cvtColor(rgbImage, this.grayImage, cv.COLOR_RGB2GRAY);

        switch (this.stage) {
            case detectionStage.DETECT_FACE: {
                this.detectFace(rgbImage, this.grayImage);
                return null;
            }
            case detectionStage.MATCH_FACE: {
                return this.distanceFromMatching(rgbImage, this.grayImage);
            }
            default: {
                throw "Cannot call this method when stage is NONE";
            }
        }
    }

    /**
     * 
     * @param {cv.Mat} rgbImage The source image.
     * @param {cv.Mat} grayImage The grayscale version of the source image.
     */
    detectFace(rgbImage, grayImage) {
        var faces = face_tracker.detectFaces(rgbImage);
        if (faces.length > 0) {
            this.stage = detectionStage.MATCH_FACE;

            var rect = faces[0];
            cv.rectangle(rgbImage, { x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y + rect.height }, [0, 255, 0, 255]);

            this.computeFocalLen(rect.height);

            let faceImg = grayImage.roi(rect);
            this.faceTmpls.init(faceImg, rgbImage.size());

            // faces.forEach(function (rect) {
            //     cv.rectangle(rgbImage, { x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y + rect.height }, [0, 255, 0, 255]);
            // });
        }
    }

    /**
     * Computes the focal length of the camera.
     * @param {number} observedFaceHgt The observed face height in pixels.
     */
    computeFocalLen(observedFaceHgt) {
        // head circumference is estimated from the person's height using the equation described in https://pubmed.ncbi.nlm.nih.gov/25552209/
        // EAR or elliptical axes ratio of the head is estimated from the chart in https://upload.wikimedia.org/wikipedia/commons/0/06/AvgHeadSizes.png
        //  MEN ABR = 20/14.5 = 1.36986301369863
        //  WOMEN ABR = 19.1/13.3 = 1.43609022556391
        // Similarly, the HWR or height-width ratio of the face is obtained by
        //  MEN HWR = 19.1/14.5 = 1.317241379310345
        //  WOMEN HWR = 17.7/13.5 = 1.311111111111111
        // SFF or shoulder from face distance is estimated at 53% of the face-to-back thickness
        //  MEN SFF = 0.106m
        //  WOMEN SFF = 0.101m
        var height = this.userHeight;
        console.log("Actual body height = %s", height);
        this.observedFaceHgt = observedFaceHgt;
        if (this.gender == genderType.MALE) {
            var circumference = (height - 70.36) / 1.734;
            console.log("Head circumference = %s", circumference.toFixed(3));
            const ear = 1.36986301369863;
            var minorAxis =
                Math.sqrt(Math.pow(0.5 * circumference / Math.PI, 2) * 2 / (ear * ear + 1)) * 2;
            console.log("Head width = %s", minorAxis.toFixed(3));
            const hwr = 1.317241379310345;
            this.actualFaceHgt = hwr * minorAxis / 100;
            console.log("Face height = %s", this.actualFaceHgt.toFixed(3));
            const sff = 0.106;
            var armLength = height / 100.0 / 2.3;
            var dist = armLength - sff;
            this.focalLen = this.observedFaceHgt * dist / this.actualFaceHgt;
        }
        else {
            var circumference = (height - 106.8) / 0.916;
            console.log("Head circumference = %s", circumference.toFixed(3));
            const ear = 1.43609022556391;
            var minorAxis =
                Math.sqrt(Math.pow(0.5 * circumference / Math.PI, 2) * 2 / (ear * ear + 1)) * 2;
            console.log("Head width = %s", minorAxis.toFixed(3));
            const hwr = 1.311111111111111;
            this.actualFaceHgt = hwr * minorAxis / 100;
            console.log("Face height = %s", this.actualFaceHgt.toFixed(3));
            const sff = 0.101;
            var armLength = height / 100.0 / 2.3;
            var dist = armLength - sff;
            this.focalLen = this.observedFaceHgt * dist / this.actualFaceHgt;
        }
    }

    /**
     * 
     * @param {cv.Mat} rgbImage The source image. 
     * @param {cv.Mat} sample The grayscale version of the source image.
     * @returns {DetectionResult} The detection result.
     */
    distanceFromMatching(rgbImage, sample) {
        var res = this.match(sample, this.faceTmpls.templates, 0.0);
        if (res != null) {
            if (res.score < 0.6) {
                console.log("distance from matching score too low.");
                return null;
            }

            if (!res.templateReference.isOriginal || res.score < 0.95) {
                var newFaceImg = sample.roi(res.box);
                this.faceTmpls.update(newFaceImg, this.smallest, this.largest, this.updateThresh);
            }

            // if (res.score < this.matchThresh) {
            //     console.log("distance from matching score lower than matching threshold.");
            //     return null;
            // }

            // draw face box
            var rect = res.box;
            cv.rectangle(rgbImage, { x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y + rect.height }, [0, 255, 0, 255]);

            res.distance = this.computeDistance(rect.height);
        }

        return res;
    }

    /**
     * Computes the distance of the face from the camera.
     * @param {number} faceHeight The height of the face in pixels.
     * @returns {number} The distance from the camera in real-world units.
     */
    computeDistance(faceHeight) {
        var d = this.actualFaceHgt * this.focalLen / faceHeight;
        switch (this.tune)
        {
            case -2:
                d *= 0.9;
                break;
            case -1:
                d *= 0.95;
                break;
            case 1:
                d *= 1.05;
                break;
            case 2:
                d *= 1.1;
                break;
        }

        return Math.min(6.5, Math.max(0, d));
    }

    /**
     * Find the matching template and use that to get the face detection result.
     * @param {cv.Mat} scene The scene to search using the templates.
     * @param {FaceTemplates} templates The templates to use for searching in the scene.
     * @param {number} thresh The min. score for a good match.
     * @returns {DetectionResult} The matching result.
     */
    match(scene, templates, thresh) {
        var res = this.getBestMatch(scene, templates, thresh);

        // if (res != null) {
            //     var s = 255.0 / res.score;
            //     var r = new cv.Mat();
            //     cv.convertScaleAbs(res.resultRaw, r, s);
            //     var heatMap = new cv.Mat();

            //     // BUG: no Javascript bindings for applyColorMap!
            //     cv.applyColorMap(r, heatMap, cv.COLORMAP_JET);
            //     let center = { x: res.box.x, y: res.box.y };
            //     let color = [0, 0, 0, 1];
            //     cv.drawMarker(heatMap, center, color);
            //     res.resultHeatMap = heatMap;
            //     r.delete();
        // }
        // else {
        //     console.log("best match returns null.");
        // }

        return res;
    }

    /**
     * Gets the best matching result.
     * @param {cv.Mat} scene The scene to search using the templates.
     * @param {FaceTemplates} templates The templates.
     * @param {number} thresh The min. score for a good match.
     * @returns {DetectionResult} The best matching result. If no face is found, returns a null.
     */
    getBestMatch(scene, templates, thresh) {
        var r = new DetectionResult();
        r.scale = 1.0;
        var j = -1;

        for (var i = 1; i < templates.length; i += 9)
            for (var k = i - 1; k <= i + 1; k++) {
                var faceTmpl = templates[k];
                var match = this.getMatchScore(scene, faceTmpl.template);
                if (this.replaceResultOnBetterScore(r, faceTmpl, match, thresh)) j = k;
            }

        var prev = j - 3;
        var next = j + 3;
        if (prev >= 0) {
            var faceTmpl = templates[prev];
            var match = this.getMatchScore(scene, faceTmpl.template);
            this.replaceResultOnBetterScore(r, faceTmpl, match, thresh);
        }

        if (next < templates.length) {
            var faceTmpl = templates[next];
            var match = this.getMatchScore(scene, faceTmpl.template);
            this.replaceResultOnBetterScore(r, faceTmpl, match, thresh);
        }

        if (r.score > 0) {
            cv.cvtColor(r.templateReference.template, r.templateRgb, cv.COLOR_GRAY2RGB);
            return r;
        }

        return null;
    }

    /**
     * Replaces existing result with a better result.
     * @param {DetectionResult} source The existing result to replace.
     * @param {TemplateType} matchTmpl The matching template.
     * @param {DetectionResult} matchRes The detection result to replace with.
     * @param {number} thresh The min. score for a good match.
     * @returns {boolean} true if the existing result is replaced with a better result; otherwise, false.
     */
    replaceResultOnBetterScore(source, matchTmpl, matchRes, thresh) {
        if (matchRes.score > thresh && matchRes.score > source.score) {
            source.score = matchRes.score;
            source.scale = matchTmpl.scale;
            source.box = matchRes.box;
            source.templateReference = matchTmpl;
            source.resultRaw.delete();
            source.resultRaw = matchRes.resultRaw;
            return true;
        }

        matchRes.dispose();
        return false;
    }

    /**
     * Gets the matching score.
     * @param {cv.Mat} scene The scene to search with the template.
     * @param {cv.Mat} template The face template.
     * @returns {DetectionResult} The matching result.
     */
    getMatchScore(scene, template) {
        var r = new DetectionResult();
        cv.matchTemplate(scene, template, r.resultRaw, cv.TM_CCOEFF_NORMED);
        cv.threshold(r.resultRaw, r.resultRaw, 0, 0, cv.THRESH_TOZERO);
        let mask = new cv.Mat();
        let minMaxRes = cv.minMaxLoc(r.resultRaw, mask);
        r.score = minMaxRes.maxVal;
        var loc = minMaxRes.maxLoc;
        var size = template.size();
        r.box = new cv.Rect(loc.x, loc.y, size.width, size.height);
        return r;
    }

    /**
     * Detects faces in the specified image.
     * @param {cv.Mat} img The image to search for faces.
     * @returns {Array} Array of faces found.
     */
    detectFaces(img) {
        var blob = cv.blobFromImage(img, 1, { width: 192, height: 144 }, [104, 117, 123, 0], false, false);
        netDet.setInput(blob);
        var out = netDet.forward();

        var faces = [];
        for (var i = 0, n = out.data32F.length; i < n; i += 7) {
            var confidence = out.data32F[i + 2];
            var left = out.data32F[i + 3] * img.cols;
            var top = out.data32F[i + 4] * img.rows;
            var right = out.data32F[i + 5] * img.cols;
            var bottom = out.data32F[i + 6] * img.rows;
            left = Math.min(Math.max(0, left), img.cols - 1);
            right = Math.min(Math.max(0, right), img.cols - 1);
            bottom = Math.min(Math.max(0, bottom), img.rows - 1);
            top = Math.min(Math.max(0, top), img.rows - 1);

            if (confidence > 0.5 && left < right && top < bottom) {
                faces.push({ x: left, y: top, width: right - left, height: bottom - top })
            }
        }
        blob.delete();
        out.delete();
        return faces;
    };
}
