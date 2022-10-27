#include "../inference/process/process.h"

int main(int argc, char* argv[]) {
  dgl::inference::ExecMasterProcess();
  return 0;
}