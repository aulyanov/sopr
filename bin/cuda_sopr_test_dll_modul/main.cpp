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

  typedef float (__stdcall *testProcedureType) (rett *a, rett *b, rett *c, const countt N);
  testProcedureType testProcedure;
  HMODULE lib = NULL;
  lib = LoadLibrary("../sopr.dll");
  if(lib){
    std::cerr << "Load libr\n";
    
    testProcedure = (testProcedureType)GetProcAddress(lib,"_matrixOnMatrixMultiply@16");

    if (testProcedure){
      std::cerr << "Find func\n";

      //INIT DATA BLOCK
      countt N = 123;

      countt B = 1;
      std::cerr << "N = " << N << endl;
      rett *x = new rett[N];
      rett *b = new rett[N];
      rett *pa = new rett[N*(B + 1)];
      rett eps = 0.00001;

      rett *a1 = new rett[N*N];
      rett *b1 = new rett[N*N];
      rett *c1 = new rett[N*N];

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
        for (int j = 0; j < N; j++){
          c1[i*N + j] = -101;
          a1[i*N + j] = 1;
          b1[i*N + j] = 1;
        }
      }

      rett * matrix = &symA[0];
      rett * smatrix = &ssymA[0];
      //END OF INIT DATA BLOCK

      cout << "start ->" << endl;
      time = testProcedure(a1,b1,c1,N);
      cout << "<- end" << endl;

      std::cerr << "timeout (ms):" << time << std::endl << std::endl;
      cout << c1[0] << "; ";
      cout << c1[N*N - 1] << "; ";
      cout << endl;
      cout << endl;
      /*
      for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
          cout << c1[i*N + j] << ";";
        }
        cout << endl;
      } */

      //TEST OUTPUT BLOCK
      /*for (int i = 0; i < N; i++){
        cout << "b:" << b[i] << "-> x:" << x[i] << endl;
      } */
      //END TEST UOTPUT BLOCK
    }
  }
  FreeLibrary(lib);
  system("PAUSE");
  return 0;
}
