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

        typedef float (__stdcall *myltiplyVectorOnScalar) (rett *a, rett *b, rett *result,const countt N);
        myltiplyVectorOnScalar myltiplyVectorOnScalarEx;

        HMODULE lib = NULL;
        lib = LoadLibrary("../sopr.dll");
	if(lib){
          std::cerr << "Load libr\n";
          myltiplyVectorOnScalarEx = (myltiplyVectorOnScalar)GetProcAddress(lib,"_myltiplyVectorOnScalar@16");

          if (myltiplyVectorOnScalarEx){
            std::cerr << "Find func\n";
            countt N = 1000000;
            std::cerr << "N = " << N << endl;

            rett *a = new rett[N];
            rett *res = new rett[N];
            rett *k = new rett();
            (*k) = -1.47;
            for (int i = 0; i < N; i++ ){
              rett p = (rett)i;
              a[i] = -2;
              res[i] = 0;
            }

            time = myltiplyVectorOnScalarEx(a,k,res,N);

            std::cerr << "timeout = " << time << "; res[N-1] = " << res[N-1] << endl;
          }
        }
        FreeLibrary(lib);
        
	system("PAUSE");
        return 0;
}

