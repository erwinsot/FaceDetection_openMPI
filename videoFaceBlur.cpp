#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h> 
#include <pthread.h>
#include <omp.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{


    int nproc; // numero de procesos
    int myrank; // id del proceso proceso
   
    Mat imgResl[301];
    MPI_Status status;
    MPI_Init(&argc,&argv);
    char msg[100];
 
    MPI_Comm_size (MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
   
    string cascadeName, nestedCascadeName;
    double scale=1;
    CascadeClassifier cascade, nestedCascade;
   
    VideoWriter writer, wrSalida;
    
    int ITERATIONS;
    int  i;
    stringstream file,file2,entradaReselsed; 
    clock_t t;
    t=clock();
    struct Video *vid;
    Mat frame, image;
    VideoCapture cap;	
    ofstream ofs("Data.txt");   
    
    cascade.load("/home/esoto/cluster/taller_parall/drive/gaze/haarcascade_frontalface_default.xml");           
    ITERATIONS=301;
    Size frame_size(600, 600);
    int fcc=cv::VideoWriter::fourcc('X','V','I','D');
    wrSalida= VideoWriter("sal_001" + std::string(".avi"),fcc,30,frame_size,true);
    if(myrank==0)
    {
     	     
      int tid =myrank;    
      int initIteration, endIteration, threadId = tid;
      initIteration = (ITERATIONS/nproc) * threadId;
      cout <<"iniciando ejecucion desde proceso: "<< tid<< "de: "<< nproc<< endl;
      if(initIteration==0)
        initIteration++;
      endIteration = initIteration + ((ITERATIONS/nproc) );
   // cout <<"desde el hilo "<< tid<< "el end es "<< endIteration<< endl;    
      vector<Rect> faces; 
      Mat gray, smallImg; 
      double fx = 1 / scale;
      stringstream entrada, salida;
    //tomamos cada frame para difuminar la parte del rostro
      while (initIteration<endIteration)
      {
        entrada<<"/home/esoto/cluster/taller_parall/frames/frame"<<initIteration<<".jpg";        
        Mat img = imread(entrada.str(), IMREAD_COLOR);
        if(img.empty())
        {
            std::cout << "Could not read the image: "<< std::endl;
            entrada.str("");
            initIteration++;
            continue;

        }
        cvtColor(img, gray, COLOR_BGR2GRAY);
        resize(gray, smallImg, Size(),fx, fx, INTER_LINEAR);
        equalizeHist(smallImg, smallImg);
        cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        for (size_t i = 0; i < faces.size(); i++)
        {
            Rect r = faces[i];
            Mat smallImgROI;
            vector<Rect> nestedObjects;
            Scalar color = Scalar(255, 0, 0); 
            // dibujar cuadrado de cara
            rectangle(img, Point(cvRound(r.x * scale), cvRound(r.y * scale)),Point(cvRound((r.x + r.width - 1) * scale),cvRound((r.y + r.height - 1) * scale)),
		 color, 3, 8, 0);

             //algoritmo para difuminar el rostro
             cv::Point topLeft = cv::Point(cvRound(r.x * scale), cvRound(r.y * scale));
             cv::Point bottomRight = cv::Point(cvRound((r.x + r.width - 1) * scale),cvRound((r.y + r.height - 1) * scale));
             cv::Rect roi = cv::Rect(topLeft, bottomRight);

             cv::GaussianBlur(img(roi), img(roi), cv::Size(91, 91), 0);
             if (nestedCascade.empty())
                continue;
             smallImgROI = smallImg(r);
        }
        salida<<"/home/esoto/cluster/taller_parall/frame_dif/frame"<<initIteration<<".jpg";
        cv::imwrite(salida.str(),img);        
        salida.str("");
        entrada.str("");
        initIteration++;



    }
      MPI_Recv(msg,13, MPI_CHAR, 1, 100, MPI_COMM_WORLD, &status);
      MPI_Recv(msg,13, MPI_CHAR, 2, 100, MPI_COMM_WORLD, &status);
      MPI_Recv(msg,13, MPI_CHAR, 3, 100, MPI_COMM_WORLD, &status);
      t = clock()-t;

      double time = (double(t)/CLOCKS_PER_SEC);
      cout << "El tiempo de ejcucion es: "<<time << endl;
      for(i=1;i<ITERATIONS;i++)
      {

        entradaReselsed<<"/home/esoto/cluster/taller_parall/frame_dif/frame"<<i<<".jpg";
        Mat im = imread(entradaReselsed.str(), IMREAD_COLOR);

        if(im.empty())
        {
            std::cout << "Could not read the image fimal: "<< std::endl;
            entradaReselsed.str("");
            
            continue;
            
        }
        imgResl[i]=im;
        
        entradaReselsed.str("");
      }

      
      for(i=1;i<sizeof imgResl;i++){
        if(imgResl[i].empty())
        {
            continue;
        }
        wrSalida.write(imgResl[i]);
     }                                   
     cout<<"Video Creado"<<endl;
}
else
{
    int tid =myrank;    
    int initIteration, endIteration, threadId = tid;
    initIteration = (ITERATIONS/nproc) * threadId;
    cout <<"iniciando ejecucion desde proceso: "<< tid<< "de: "<< nproc<< endl;
    if(initIteration==0)
        initIteration++;
    endIteration = initIteration + ((ITERATIONS/nproc) );
    //cout <<"desde el hilo "<< tid<< "el end es "<< endIteration<< endl;    
    vector<Rect> faces; 
    Mat gray, smallImg; 
 double fx = 1 / scale;
    stringstream entrada, salida;
    //tomamos cada frame para difuminar la parte del rostro
    while (initIteration<endIteration)
    {
        entrada<<"/home/esoto/cluster/taller_parall/frames/frame"<<initIteration<<".jpg";        
        Mat img = imread(entrada.str(), IMREAD_COLOR);
        if(img.empty())
        {
            std::cout << "Could not read the image: "<< std::endl;
            entrada.str("");
            initIteration++;
            continue;

        }
        cvtColor(img, gray, COLOR_BGR2GRAY);
        resize(gray, smallImg, Size(),fx, fx, INTER_LINEAR);
        equalizeHist(smallImg, smallImg);
        cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        for (size_t i = 0; i < faces.size(); i++)
        {
            Rect r = faces[i];
            Mat smallImgROI;
            vector<Rect> nestedObjects;
            Scalar color = Scalar(255, 0, 0); 


            // dibujar cuadrado de cara
             rectangle(img, Point(cvRound(r.x * scale), cvRound(r.y * scale)),Point(cvRound((r.x + r.width - 1) * scale),cvRound((r.y + r.height - 1) * scale)), color, 3, 8, 0);
             //algoritmo para difuminar el rostro
             cv::Point topLeft = cv::Point(cvRound(r.x * scale), cvRound(r.y * scale));
             cv::Point bottomRight = cv::Point(cvRound((r.x + r.width - 1) * scale),cvRound((r.y + r.height - 1) * scale));
             cv::Rect roi = cv::Rect(topLeft, bottomRight);

             cv::GaussianBlur(img(roi), img(roi), cv::Size(91, 91), 0);
             if (nestedCascade.empty())
                continue;
             smallImgROI = smallImg(r);
        }
        salida<<"/home/esoto/cluster/taller_parall/frame_dif/frame"<<initIteration<<".jpg";
        cv::imwrite(salida.str(),img);        
        salida.str("");
        entrada.str("");
        initIteration++;

  }
        strcpy(msg,"Hello World!");
 	 MPI_Send(msg,13, MPI_CHAR, 0, 100, MPI_COMM_WORLD);
} 
   MPI_Finalize ();

    return 0;


}
