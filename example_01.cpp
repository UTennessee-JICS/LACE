/*

#mac osx
g++ -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc
ar -rv libgtest.a gtest-all.o
g++ -std=c++11 -isystem ${GTEST_DIR}/include -pthread example_01.cpp libgtest.a -o example_01

#beacon
icpc -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc
ar -rv libgtest.a gtest-all.o
icpc -std=c++11 -isystem ${GTEST_DIR}/include -pthread example_01.cpp libgtest.a -o example_01

./example_01 --gtest_output="xml:report.xml"

*/

#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>

struct BankAccount
{
  int balance = 0;

  BankAccount()
  {
  }

  explicit BankAccount(const int balance)
    : balance(balance)
  {
  }

  void deposit(int amount)
  {
    balance += amount;
  }

  bool withdraw(int amount)
  {
    if(amount <= balance)
    {
      balance -= amount;
      return true;
    }
    return false;
  }
};

struct BankAccountTest : testing::Test
{
  BankAccount* account;

  BankAccountTest()
  {
    account = new BankAccount;
  }

  virtual ~BankAccountTest()
  {
    delete account;
  }
};

TEST_F(BankAccountTest, BankAccountStartsEmpty)
{
  EXPECT_EQ(0, account->balance);
}

TEST_F(BankAccountTest, CanDepositMoney)
{
  account->deposit(100);
  EXPECT_EQ(100, account->balance);
}

struct account_state
{
  int initial_balance;
  int withdrawal_amount;
  int final_balance;
  bool success;

  friend std::ostream& operator <<(std::ostream& os, const account_state& obj)
  {
    return os
      << "initial_balance: " << obj.initial_balance
      << " withdrawal_amount: " << obj.withdrawal_amount
      << " final_balance: " << obj.final_balance
      << " success: " << obj.success;
  }
};

struct WithdrawAccountTest : BankAccountTest, testing::WithParamInterface<account_state>
{
  WithdrawAccountTest()
  {
    account->balance = GetParam().initial_balance;
  }
};

TEST_P(WithdrawAccountTest, FinalBalance)
{
  auto as = GetParam();
  auto success = account->withdraw(as.withdrawal_amount);
  EXPECT_EQ(as.final_balance, account->balance);
  EXPECT_EQ(as.success, success);
}

INSTANTIATE_TEST_CASE_P(Default, WithdrawAccountTest,
  testing::Values(
    account_state{100, 50, 50, true},
    account_state{100, 200, 100, false}
    ));

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
