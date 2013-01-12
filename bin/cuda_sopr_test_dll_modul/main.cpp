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

        typedef float (__stdcall *matrixOnVectorMultiply) (rett *A, rett *b, rett *result,countt N);
        matrixOnVectorMultiply matrixOnVectorMultiplyEx;

        HMODULE lib = NULL;
        lib = LoadLibrary("../sopr.dll");
	if(lib){
          std::cerr << "Load libr\n";
          matrixOnVectorMultiplyEx = (matrixOnVectorMultiply)GetProcAddress(lib,"_matrixOnVectorMultiply@16");

          if (matrixOnVectorMultiplyEx){
            std::cerr << "Find func\n";

//for (countt p = 17929; p < 17934; p+=1 ){
for (countt p = 18000; p < 19000; p+=100 ){  // out of memory
        countt N = p;
        std::cerr << N << endl;
        rett *A = new rett[N*N];
        rett *b = new rett[N];
        rett *result = new rett[N];

        rett cons = rand();
        for (countt i = 0; i < N; i++){

          b[i] = cons;
          result[i] = 0;
          for (countt j = 0; j < N; j++){
            A[i*N + j] = 1;
          }
        }

            time = matrixOnVectorMultiplyEx(A,b,result,N);
            std::cerr << "exec time(milliseconds):" << time << "; result[0]=" << result[0] << "; result[N-1]=" << result[N-1] << endl;
        delete [] A;
        delete [] b;
        delete [] result;
}
          }
        }
        FreeLibrary(lib);
        
	system("PAUSE");
        return 0;
}

