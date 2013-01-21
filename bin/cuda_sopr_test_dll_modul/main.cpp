//---------------------------------------------------------------------------

#include <vcl.h>
#include <iostream>
#include <Math.h>
#include <stdlib.h>


#pragma hdrstop

#include "../../GlobalEntitys.h"
//---------------------------------------------------------------------------

#pragma argsused

using namespace std;

int main(int argc, char* argv[]) {

        rett symAo[7*7] = {
		9.0 ,  0.0 , 0.0 , 3.0 , 1.0 , 0.0 , 1.0 ,
		0.0 ,  11.0, 2.0 , 1.0 , 5.0 , 0.0 , 2.0 ,
		0.0 ,  2.0 , 10.0 , 2.0 , -7.0 , 0.0 , 0.0 ,
		3.0 ,  1.0 ,  2.0 ,  9.0 , 1.0 , 0.0 , 0.0 ,
		1.0 ,  5.0 ,  -7.0,  1.0 ,  12.0 , 0.0 , 1.0 ,
		0.0 ,  0.0 ,  0.0 ,  0.0 ,  0.0  , 8.0 ,  0.0 ,
		1.0 ,  2.0 ,  0.0 ,  0.0 ,  1.0  , 0.0 ,  80.0
	};

        rett symA[7*7] = {
		9.0 ,  0.0 , 0.0  ,  0.0 ,   0.0 , 0.0 ,  0.0 ,
	        0.0 ,  11.0, 2.0  ,  1.0 ,   0.0 , 0.0 ,  0.0 ,
		0.0 ,  2.0 , 10.0 ,  2.0 ,  -7.0 , 0.0 ,  0.0 ,
		0.0 ,  1.0 ,  2.0 ,  9.0 ,  1.0  , 0.0 ,  0.0 ,
		0.0 ,  0.0 , -7.0 ,  1.0 ,  12.0 , 0.0 ,  0.0 ,
		0.0 ,  0.0 ,  0.0 ,  0.0 ,   0.0 , 8.0 ,  0.0 ,
		0.0 ,  0.0 ,  0.0 ,  0.0 ,   0.0 , 0.0 , 80.0
	};

        rett ssymA[7*3] = {
		0.0 ,  0.0 , 9.0 ,
                0.0 ,  0.0 , 11.0 ,
                0.0 ,  2.0 , 10.0 ,
                1.0 ,  2.0 , 9.0 ,
               -7.0 ,  1.0 , 12.0 ,
                0.0 ,  0.0 , 8.0 ,
                0.0 ,  0.0 , 80.0
	};

        float time = 0;

        typedef float (__stdcall *methodConjugateGradientForStripMatrix) (rett *matrixA, rett *vectorB, rett *vectorX, const countt N, rett eps);
        methodConjugateGradientForStripMatrix methodConjugateGradientForStripMatrixEx;

        HMODULE lib = NULL;
        lib = LoadLibrary("../sopr.dll");
	if(lib){
          std::cerr << "Load libr\n";
          methodConjugateGradientForStripMatrixEx = (methodConjugateGradientForStripMatrix)GetProcAddress(lib,"_methodConjugateGradient@20");

          if (methodConjugateGradientForStripMatrixEx){
            std::cerr << "Find func\n";

int o;
            countt N = 7;
//            countt N = 1000;
            countt B = 2;
            std::cerr << "N = " << N << endl;

            rett *x = new rett[N];
            rett *b = new rett[N];
            rett eps = 0.00001;
            for (int i = 0; i < N; i++ ){
              x[i] = 1;
              b[i] = N/(i + 1);
            }
//std::cin >> o;
            rett * matrix = &symAo[0];
            time = methodConjugateGradientForStripMatrixEx(matrix,b,x,N, eps);
            time = methodConjugateGradientForStripMatrixEx(matrix,b,x,N, eps);
            time = methodConjugateGradientForStripMatrixEx(matrix,b,x,N, eps);
//std::cin >> o;
            std::cerr << "timeout (ms):" << time << std::endl << std::endl;
            for (int i = 0; i < N; i++){
              rett p = 0;
              for (int j = 0; j < N; j++){
                p += matrix[i*N + j ]*x[j];
              }
              std::cerr << "b[ " << i << " ] = " << b[i] << " ; b'[ " << i << " ] = " << p << "; x = << " << x[i] << "\n";
            }
          }
        }
        FreeLibrary(lib);
        
	system("PAUSE");
        return 0;
}

