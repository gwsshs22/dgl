#include "../inference/entrypoint.h"

#include <sys/prctl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <string>
#include <iostream>
#include <thread>
#include <string.h>

void child_main() {
  std::string test;
  std::cout << "I am a child! (pid=" << getpid() << ")" << std::endl;
  std::cin >> test;
  std::cout << "Child read this: "<< test << std::endl;
  std::this_thread::sleep_for(std::chrono::minutes(1));
}

void parent_main() {

  pid_t pid = fork();
  if (pid == 0) {
    // Launch a child
    setenv("IS_CHILD", "TRUE", 1);
    char exe[1024];
    int ret;

    ret = readlink("/proc/self/exe", exe, sizeof(exe)-1);
    exe[ret] = 0;
    if(ret ==-1) {
        fprintf(stderr,"ERRORRRRR\n");
        exit(1);
    }
    std::cerr << "Try " << exe << std::endl;
    ret = execlp(exe, exe, (char *)NULL);
    if (ret < 0) {
      std::cerr << "Unexpected: " << strerror(errno) << std::endl;
      exit(-1);
    }
    // never reached
  }
  
  // Parent code
  std::cout << "I am a parent! (pid=" << getpid() << ")" << std::endl;
  
}

int main(int argc, char* argv[]) {
  std::cout << argv[0] << std::endl;
  std::string is_child = "";
  const char* env_val = getenv("IS_CHILD");
  if (env_val != NULL) {
    is_child = env_val;
  }

  if (is_child.compare("TRUE") == 0) {
    child_main();
  } else {
    parent_main();
  }

  return 0;
}
