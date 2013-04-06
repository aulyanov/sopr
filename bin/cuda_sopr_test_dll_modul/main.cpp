//---------------------------------------------------------------------------

#include <vcl.h>
#include <iostream>
#include <Math.h>
#include <stdlib.h>
#include "windows.h"


#pragma hdrstop

#include "../../GlobalEntitys.h"
//#include "../../cuda_math.h"
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
  rett ssymA[] = {
		0.0 ,  0.0 , 9.0 ,
                0.0 ,  0.0 , 11.0 ,
                0.0 ,  2.0 , 10.0 ,
                1.0 ,  2.0 , 9.0 ,
               -7.0 ,  1.0 , 12.0 ,
                0.0 ,  0.0 , 8.0 ,
                0.0 ,  0.0 , 80.0
  };

  float time = 0;

  typedef float (__stdcall *testProcedureType) (rett *matrixA, rett *vectorB, rett *vectorX, const countt N, const countt B, rett eps);
  testProcedureType testProcedure;
  HMODULE lib = NULL;
  lib = LoadLibrary("../sopr.dll");
  if(lib){
    std::cerr << "Load libr\n";
    
    testProcedure = (testProcedureType)GetProcAddress(lib,"_methodConjugateGradientForBandMatrix@24");
    if (testProcedure){
      std::cerr << "Find func\n";

      //INIT DATA BLOCK
      for (countt NN = 100; NN < 3100; NN+=100){

      
      countt N = NN;

      countt B = 1;
      //std::cerr << "N = " << N << endl;
      rett *x = new rett[N];
      rett *b = new rett[N];
      rett *pa = new rett[N*(B + 1)];
      rett eps = 0.00001;

      for (int i = 0; i < N; i++ ){
        if (i < 1){
          pa[0] = 0;
          pa[1] = 2;
        }else{
          pa[i*(B+1)] = -1;
          pa[i*(B+1)+1] = 2;
        }
        //cout << "pa0:" << pa[i*(B+1)] << " pa1:" << pa[i*(B+1)+1] << endl;
        x[i] = 0.5;
        b[i] = N/(i + 1);
      }
      //END OF INIT DATA BLOCK

      time = testProcedure(pa,b,x,N,B,eps);

      std::cerr << N << "->" << time << std::endl;
      delete [] pa;
      delete [] b;
      delete [] x;
      
      }
    }
  }
  FreeLibrary(lib);
  system("PAUSE");
  return 0;
}
