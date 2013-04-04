//---------------------------------------------------------------------------

#include <vcl.h>
#include <iostream>
#include <Math.h>
#include <stdlib.h>
#include "windows.h"

#include <stdio.h>
#include <conio.h>
#include <time.h>

#pragma hdrstop

#include "../../GlobalEntitys.h"
//#include "../../cuda_math.h"
//---------------------------------------------------------------------------

#pragma argsused

using namespace std;

int main(int argc, char* argv[]) {

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
      for (int NN = 100; NN < 10000; NN+=100){
      countt N = NN;//1000;//10100;
      
      rett *a1 = new rett[N*N];
      rett *b1 = new rett[N*N];
      rett *c1 = new rett[N*N];

      for (int i = 0; i < N; i++ ){
        for (int j = 0; j < N; j++){
          c1[i*N + j] = -101;
          a1[i*N + j] = 1;
          b1[i*N + j] = 1;
        }
      }
      //-----------------------
unsigned long t=clock();
      cout << "N = " << N << "; CPU:";
      for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j ++){
          c1[i*N + j] = 0;
          for (int k = 0; k < N; k++){
            c1[i*N + j] += a1[i*N + k]*b1[k*N + j];
          }
        }
      }
      time = clock()-t;
      std::cerr << "timeout (ms):" << time << "; ";

      cout << "GPU:";
        time = testProcedure(a1,b1,c1,N);
      std::cerr << "timeout (ms):" << time << std::endl;
      delete [] a1;
      delete [] b1;
      delete [] c1;
      }
      /*
      for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
          cout << c1[i*N + j] << ";";
        }
        cout << endl;
      } */
    }
  }
  FreeLibrary(lib);
  system("PAUSE");
  return 0;
}
