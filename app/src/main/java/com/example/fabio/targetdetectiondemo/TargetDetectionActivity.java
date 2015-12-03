package com.example.fabio.targetdetectiondemo;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.Toast;
import android.widget.ToggleButton;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Demo for target detection (bright-colored ball) using opencv.
 *
 * @author fiorfe01 - Fabio Fiori
 * @author - Isaac Smith
 */
public class TargetDetectionActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private Mat rgbaMat; // original image matrix
    private Mat pyrDownMat; // downsized image matrix
    private Mat hsvMat; // hsv image matrix
    private Mat binaryMat; // binary image matrix
    private Mat denoisedMat; // denoised image matrix
    private Mat hierarchyMat; // hierarchy image matrix

    // lower and upper threshold for hsv target color
    private Scalar lowerThreshold, upperThreshold;

    private Scalar CONTOUR_COLOR; // contour color

    // screen size variables
    private int mCameraViewWidth;
    private int mCameraViewHeight;

    private boolean follow; // follow target variable

    private Rect targetHSVRect; // target hsv rectangle

    private Robot rob; // Robot obj for communication

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.color_blob_detection_activity_surface_view);

        ToggleButton toggle = (ToggleButton) findViewById(R.id.btnClick);

        toggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {

                follow = isChecked;
            }
        });

        Button btnDetect = (Button) findViewById(R.id.btnDetectColor);

        btnDetect.setOnClickListener(new View.OnClickListener() {

            public void onClick(View v) {

                getHSV();
            }
        });

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);

        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {

        super.onPause();

        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {

        super.onResume();

        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
    }

    public void onDestroy() {

        super.onDestroy();

        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        // instantiate matrices
        rgbaMat = new Mat(height, width, CvType.CV_8UC4);
        pyrDownMat = new Mat(height, width, CvType.CV_8UC4);
        hsvMat = new Mat(height, width, CvType.CV_8UC4);
        binaryMat = new Mat(height, width, CvType.CV_8UC4);
        denoisedMat = new Mat(height, width, CvType.CV_8UC4);
        hierarchyMat = new Mat(height, width, CvType.CV_8UC4);

        lowerThreshold = new Scalar(-40, 110, 170); // lower target hsv values (orange)
        upperThreshold = new Scalar(24, 261, 256); // upper target hsv values (orange)

        CONTOUR_COLOR = new Scalar(0, 255, 0, 255);

        // record the screen size constants
        mCameraViewWidth = width;
        mCameraViewHeight = height;

        // instantiate target hsv rectangle
        targetHSVRect = new Rect();

        targetHSVRect.width = mCameraViewWidth / 16;
        targetHSVRect.height = mCameraViewHeight / 16;

        targetHSVRect.x = mCameraViewWidth / 2 - targetHSVRect.width;
        targetHSVRect.y = mCameraViewHeight / 2 + targetHSVRect.height;

        rob = new Robot("192.168.1.3");
    }

    @Override
    public void onCameraViewStopped() {

        rgbaMat.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // original image capture
        rgbaMat = inputFrame.rgba();

        if (follow) {

            followTarget();

        } else {

            // draw rectangle
            Imgproc.rectangle(rgbaMat, new Point(targetHSVRect.x, targetHSVRect.y),
                    new Point(targetHSVRect.x + 2 * targetHSVRect.width, targetHSVRect.y - 2 * targetHSVRect.height), CONTOUR_COLOR, 5);
        }

        return rgbaMat;
    }

    /**
     * Gets HSV from target hsv rectangle.
     */
    private void getHSV() {

        Mat targetRegionRgba = rgbaMat.submat(targetHSVRect);

        Mat targetRegionHsv = new Mat();

        Imgproc.cvtColor(targetRegionRgba, targetRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        // calculate average color of touched region
        Scalar hsv = Core.sumElems(targetRegionHsv);

        int pointCount = targetHSVRect.width * targetHSVRect.height;

        for (int i = 0; i < hsv.val.length; i++)
            hsv.val[i] /= pointCount;

        setHsvRange(hsv);
    }

    /**
     * Sets the HSV range.
     *
     * @param hsvColor
     */
    public void setHsvRange(Scalar hsvColor) {

        Scalar mColorRadius = new Scalar(25, 50, 50, 0);

        // ensures that the hue adjusted hue range is between 0 and 255
        double minH = (hsvColor.val[0] >= mColorRadius.val[0]) ? hsvColor.val[0] - mColorRadius.val[0] : 0;
        double maxH = (hsvColor.val[0] + mColorRadius.val[0] <= 255) ? hsvColor.val[0] + mColorRadius.val[0] : 255;

        lowerThreshold.val[0] = minH;
        upperThreshold.val[0] = maxH;

        lowerThreshold.val[1] = hsvColor.val[1] - mColorRadius.val[1];
        upperThreshold.val[1] = hsvColor.val[1] + mColorRadius.val[1];

        lowerThreshold.val[2] = hsvColor.val[2] - mColorRadius.val[2];
        upperThreshold.val[2] = hsvColor.val[2] + mColorRadius.val[2];

        lowerThreshold.val[3] = 0;
        upperThreshold.val[3] = 255;

        // output hsv range
        Toast.makeText(this, "LOWER HSV = " + lowerThreshold, Toast.LENGTH_LONG).show();
        Toast.makeText(this, "UPPER HSV = " + upperThreshold, Toast.LENGTH_LONG).show();
    }

    /**
     * Follows the target.
     */
    private void followTarget() {

        // downsize original image (divide length and height by 4) for faster image processing
        Imgproc.pyrDown(rgbaMat, pyrDownMat);
        Imgproc.pyrDown(pyrDownMat, pyrDownMat);

        // convert downgraded image to hsv
        Imgproc.cvtColor(pyrDownMat, hsvMat, Imgproc.COLOR_RGB2HSV_FULL);

        // convert hsv to binary image
        Core.inRange(hsvMat, lowerThreshold, upperThreshold, binaryMat);

        // dilate and erode binary image
        Imgproc.dilate(binaryMat, denoisedMat, new Mat());
        Imgproc.erode(denoisedMat, denoisedMat, new Mat());

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        // contours determination
        Imgproc.findContours(denoisedMat, contours, hierarchyMat, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        double largestArea = 0.0;

        MatOfPoint largestContour = new MatOfPoint();

        // iterate through each contour
        for (MatOfPoint currContour : contours) {

            // find the area of contour
            double contourArea = Imgproc.contourArea(currContour);

            if (contourArea > largestArea) {

                largestArea = contourArea;

                // store the largest contour
                largestContour = currContour;
            }
        }

        // if target is found
        if (largestArea != 0) {

            // resize largest contour to fit the original image size
            Core.multiply(largestContour, new Scalar(4, 4), largestContour);

            MatOfPoint2f convLargestContour = new MatOfPoint2f();

            // get largest contour
            largestContour.convertTo(convLargestContour, CvType.CV_32FC2);

            Point center = new Point();
            float[] radius = new float[1];

            // get target center and radius
            Imgproc.minEnclosingCircle(convLargestContour, center, radius);

            // draw perimeter
            Imgproc.circle(rgbaMat, center, Math.round(radius[0]), CONTOUR_COLOR, 7);

            // draw center
            Imgproc.circle(rgbaMat, center, 2, CONTOUR_COLOR, 7);

            // send command to robot unit
            rob.drive(150, (360 * (1 - center.x / mCameraViewWidth)));

        } else {

            rob.stop();
        }
    }
}