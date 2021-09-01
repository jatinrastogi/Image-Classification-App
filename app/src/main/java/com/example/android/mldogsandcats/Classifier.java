package com.example.android.mldogsandcats;

import android.graphics.Bitmap;

import androidx.annotation.NonNull;

import java.text.DecimalFormat;
import java.util.List;


public interface Classifier {

    class Recognition
    {

        private final Float confidence;

        public Recognition(final Float confidence)
        {
            this.confidence = confidence;
        }

        @NonNull
        @Override
        public String toString()
        {
            DecimalFormat df = new DecimalFormat("0.00");
            String resultString = "";

            if (confidence != null) {
                float cat = (100 *(1- confidence));
                float dog = 100*confidence;
                String C = df.format(cat);
                String D = df.format(dog);
                resultString += "This image is "+C+"% cat and "+D+"% dog";
            }

            return resultString.trim();
        }
    }


    List<Recognition> recognizeImage(Bitmap bitmap);

    void close();
}
