
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "mmio.h"
#include "sparse_types.h"
#include "container_tests.h"

int my_argc;
char** my_argv;

int main(int argc, char* argv[])
{
  printf("\nLACE\nUnit testing is fun-da-mental.\n\n");
  testing::InitGoogleTest(&argc, argv);
  my_argc = argc;
  my_argv = argv;
  return RUN_ALL_TESTS();

}
