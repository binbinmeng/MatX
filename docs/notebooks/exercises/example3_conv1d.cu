#include <matx.h>

using namespace matx;

int main() {

  tensor_t<float, 1> C({16});
  tensor_t<float, 1> filt({3});
  tensor_t<float, 1> Co({16 + filt.Lsize() - 1});

  filt = {1.0/3, 1.0/3, 1.0/3};

  randomGenerator_t<float> randData(C.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<1>({16}, NORMAL);
  (C = randTensor1).run();  

  printf("Initial C tensor:\n");
  C.Print();

  // TODO: Perform a 1D direct convolution on C with filter filt
  

  printf("After conv1d:\n");
  Co.Print();

  return 0;
}
