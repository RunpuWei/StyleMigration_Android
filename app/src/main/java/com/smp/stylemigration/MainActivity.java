package com.smp.stylemigration;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity implements Runnable{
    private ImageView mImageView;
    private Button mButtonSegment;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            // 读取assets文件夹内的image.jpg文件
            // app/src/main/assets/image.jpg
            // 默认设置的图片分辨率应为512×512，如果转换的太慢可以稍微减小一丢丢！
            mBitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
            mBitmap = changeBitmapSize(mBitmap,512,512);
        } catch (IOException e) {
            Log.e("DepthEstimation", "Error reading assets", e);
            finish();
        }

        // 显示原图片
        mImageView = findViewById(R.id.image);
        mImageView.setImageBitmap(mBitmap);

        try {
            // 读取assets文件夹内的LandscapeModel.ptl模型文件
            // app/src/main/assets/image.jpg
            mModule = Module.load(assetFilePath(this, "LandscapeModel.ptl"));
        } catch (IOException e) {
            Log.e("DepthEstimation", "Error reading assets", e);
            finish();
        }


        mProgressBar = findViewById(R.id.progressBar);
        mButtonSegment = findViewById(R.id.segmentButton);
        mButtonSegment.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText("转换ing");

                //开始图片转换线程
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

    }

    //图片转换线程
    @Override
    public void run() {
        //创建模型输入部分
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final float[] inputs = inputTensor.getDataAsFloatArray();


        //模型运行部分
        final long startTime = SystemClock.elapsedRealtime();
        IValue outTensors = mModule.forward(IValue.from(inputTensor));
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("DepthEstimation",  "inference time (ms): " + inferenceTime);
        System.out.println(inferenceTime);


        //模型输出部分
        final Tensor outputTensor = outTensors.toTensor();
        final float[] intValues = outputTensor.getDataAsFloatArray();


        //输出图像的显示部分
        int width = mBitmap.getWidth();
        int height = mBitmap.getHeight();
        ArrayList<Float> arraylist = new ArrayList<>();
        for (int i = 0 ; i< intValues.length ; i++){
            arraylist.add(intValues[i]);
        }
        final Bitmap bitmap = arrayFlotToBitmap(arraylist, width, height);
        runOnUiThread(() -> {
            mImageView.setImageBitmap(bitmap);
            mButtonSegment.setEnabled(true);
            mButtonSegment.setText("转换完成");
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
        });
    }

    /**
     * FloatArray转Bitmap位图
     *
     * @return Bitmap 位图图像
     */
    private static Bitmap arrayFlotToBitmap(List<Float> floatArray, int width, int height){

        byte alpha = (byte) 255 ;

        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888) ;

        ByteBuffer byteBuffer = ByteBuffer.allocate(width*height*4*3) ;

        float Maximum = Collections.max(floatArray);
        float minmum = Collections.min(floatArray);
        float delta = Maximum - minmum ;

        int i = 0 ;
        for (float value : floatArray){
            byte temValue = (byte) ((byte) ((((value-minmum)/delta)*255)));
            byteBuffer.put(4*i, temValue) ;
            byteBuffer.put(4*i+1, temValue) ;
            byteBuffer.put(4*i+2, temValue) ;
            byteBuffer.put(4*i+3, alpha) ;
            i++ ;
        }
        bmp.copyPixelsFromBuffer(byteBuffer) ;
        return bmp ;
    }

    /**
     * 将文件复制到/app/src/main/assets目录后，输入文件名，返回此文件在Android内的绝对路径。
     *
     * @return absolute file path 文件绝对路径
     */
    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    /**
     * 位图尺寸修改方法
     *
     * @return absolute file path 文件绝对路径
     */
    private Bitmap changeBitmapSize(Bitmap bitmap,int newWidth,int newHeight) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Log.e("width","width:"+width);
        Log.e("height","height:"+height);


        //计算压缩的比率
        float scaleWidth=((float)newWidth)/width;
        float scaleHeight=((float)newHeight)/height;

        //获取想要缩放的matrix
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth,scaleHeight);

        //获取新的bitmap
        bitmap=Bitmap.createBitmap(bitmap,0,0,width,height,matrix,true);
        bitmap.getWidth();
        bitmap.getHeight();
        Log.e("newWidth","newWidth"+bitmap.getWidth());
        Log.e("newHeight","newHeight"+bitmap.getHeight());
        return bitmap;
    }

}
