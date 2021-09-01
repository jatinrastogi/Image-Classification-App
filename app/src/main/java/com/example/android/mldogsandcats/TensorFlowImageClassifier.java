package com.example.android.mldogsandcats;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class TensorFlowImageClassifier implements Classifier
{
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;

    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;

    private Interpreter interpreter;
    private int inputSize;

    private TensorFlowImageClassifier() {

    }

    static Classifier create(AssetManager assetManager,
                             String modelPath,
                             int inputSize) throws IOException {

        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath), new Interpreter.Options());
        classifier.inputSize = inputSize;

        return classifier;
    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap)
    {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float [][] result = new float[1][1];
        interpreter.run(byteBuffer, result);
        return getSortedResultFloat(result);
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException
    {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap)
    {
        ByteBuffer byteBuffer;
        byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * BATCH_SIZE * PIXEL_SIZE);

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i)
        {
            for (int j = 0; j < inputSize; ++j)
            {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
        return byteBuffer;
    }


    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResultFloat(float[][] labelProbArray)
    {
        final ArrayList<Recognition> recognitions = new ArrayList<>();
        recognitions.add(new Recognition(labelProbArray[0][0]));

        return recognitions;
    }

}
