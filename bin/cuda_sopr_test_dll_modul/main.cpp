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

        rett symA[7*7] = {
		9.0 ,  0.0 , 0.0 , 3.0 , 1.0 , 0.0 , 1.0 ,
		0.0 ,  11.0, 2.0 , 1.0 , 5.0 , 0.0 , 2.0 ,
		0.0 ,  2.0 , 10.0 , 2.0 , -7.0 , 0.0 , 0.0 ,
		3.0 ,  1.0 ,  2.0 ,  9.0 , 1.0 , 0.0 , 0.0 ,
		1.0 ,  5.0 ,  -7.0,  1.0 ,  12.0 , 0.0 , 1.0 ,
		0.0 ,  0.0 ,  0.0 ,  0.0 ,  0.0  , 8.0 ,  0.0 ,
		1.0 ,  2.0 ,  0.0 ,  0.0 ,  1.0  , 0.0 ,  80.0
	};

        float time = 0;

        //_matrixOnVectorMultiply@16

        typedef float (__stdcall *matrixOnVectorMultiply) (rett *A, rett *b, rett *result,countt N, countt partNumber, countt partSize);
        matrixOnVectorMultiply matrixOnVectorMultiplyEx;

        HMODULE lib = NULL;
        lib = LoadLibrary("../sopr.dll");
	if(lib){
          std::cerr << "Load libr\n";
          matrixOnVectorMultiplyEx = (matrixOnVectorMultiply)GetProcAddress(lib,"_matrixPartOnVectorMultiply@24");

          if (matrixOnVectorMultiplyEx){
            std::cerr << "Find func\n";


        countt count = 20;
        countt partSize = 1024;
        countt N = count*partSize;
        std::cerr << N << endl;
        rett *A = new rett[N*partSize];
        rett *b = new rett[N];
        rett *result = new rett[N];
        
        for (countt i = 0; i < N; i++){
          b[i] = -1;
          result[i] = 0;
        }
        for (int i = 0; i < partSize; i++){
          for (countt j = 0; j < N; j++){
            A[i*N + j] = 1;
          }
        }
for (int pn = 0; pn < count; pn++){

        time = matrixOnVectorMultiplyEx(A,b,result,N,pn,partSize);
        int ppp = (pn + 1)*partSize + 1;
        std::cerr << "exec time(milliseconds):" << time << "; " << result[0] << "; " << result[partSize*(pn + 1 ) - 1] << "; " << result[N-1] << endl;
}
        delete [] A;
        delete [] b;
        delete [] result;

          }
        }
        FreeLibrary(lib);
        
	system("PAUSE");
        return 0;
}

