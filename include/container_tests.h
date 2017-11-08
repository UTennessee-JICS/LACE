#ifndef DEV_CONTAINERTESTS_H
#define DEV_CONTAINERTESTS_H

#define EXPECT_ARRAY_INT_EQ( length, ref, target) \
{ \
  int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}

#define EXPECT_ARRAY_DOUBLE_EQ( length, ref, target) \
{ \
  int i = 0; \
  for(i=0; i<length; i++) { \
    EXPECT_DOUBLE_EQ(ref[i], target[i]) \
    << "Arrays " #ref  " and " #target \
           "differ at index " << i; \
  } \
}

#endif
