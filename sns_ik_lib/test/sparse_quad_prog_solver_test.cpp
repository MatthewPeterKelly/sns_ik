/*! \file sparse_quad_prog_solver_test.cpp
 * \brief Unit Test: sparse_quad_prog_solver_test
 * \author Matthew Kelly
 */
/*
 *    Copyright 2018 Rethink Robotics
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <ros/console.h>

#include "rng_utilities.hpp"
#include "sparse_quad_prog_solver.hpp"

/*************************************************************************************************/

TEST(sparse_quad_prog_solver, HelloWorld)
{
  // Very simple test problem, no constraints, solution at origin.
  sns_ik::rng_util::setRngSeed(83907, 17302);
  int nVar = 2;
  int nCst = 1;
  int maxIter = 5;
  double tol = 1e-10;
  Eigen::VectorXd xLow = sns_ik::rng_util::getRngVectorXd(0, nVar, -1.0, 0.1);
  Eigen::VectorXd xUpp = sns_ik::rng_util::getRngVectorXd(0, nVar, 0.1, 1.0);
  Eigen::VectorXd w = sns_ik::rng_util::getRngVectorXd(0, nVar, 0.1, 1.0);
  Eigen::MatrixXd A(nCst, nVar);  A << 1.0, 1.0;
  Eigen::VectorXd b(nCst); b << 0.0;

  // Solve:
  sns_ik::SparseQuadProgSolver solver;
  sns_ik::SparseQuadProgSolver::Result result = solver.solve(xLow, xUpp, w, A, b, maxIter, tol);
  ASSERT_NEAR(result.objVal, 0.0, 1e-8);
}

/*************************************************************************************************/

TEST(sparse_quad_prog_solver, SimpleTestOne)
{
  // // MATLAB CODE:
  // H = diag([1,3,2]);
  // zLow = [-1; -2; -3];
  // zUpp = [ 2;  5;  4];
  // Aeq = [ 1, 2, 2;
  //        -2, 3, 5];
  // beq = [2; -5];
  // f = [0;0;0]; A = []; b = [];
  // [z, objVal, exitCode] = quadprog(H,f,A,b,Aeq,beq,zLow,zUpp);
  // >> z = [2.0; 0.5; -0.5];  // solution decision variables
  // >> objVal = 2.625;  // solution objective function value

  int maxIter = 5;
  double tol = 1e-10;
  Eigen::VectorXd xLow(3); xLow << -1.0, -2.0, -3.0;
  Eigen::VectorXd xUpp(3); xUpp << 2.0, 5.0, 4.0;
  Eigen::VectorXd w(3); w << 1.0, 3.0, 2.0;
  Eigen::MatrixXd A(2, 3);
  A.row(0) << 1.0, 2.0, 2.0;
  A.row(1) << -2.0, 3.0, 5.0;
  Eigen::VectorXd b(2); b << 2.0, -5.0;

  // Solve:
  sns_ik::SparseQuadProgSolver solver;
  sns_ik::SparseQuadProgSolver::Result result = solver.solve(xLow, xUpp, w, A, b, maxIter, tol);
  tol = 1e-6;
  ASSERT_NEAR(result.objVal, 2.625, tol);
  ASSERT_NEAR(result.soln(0), 2.0, tol);
  ASSERT_NEAR(result.soln(1), 0.5, tol);
  ASSERT_NEAR(result.soln(2), -0.5, tol);
}

/*************************************************************************************************/

/*
 * This test fails due to a problem with the active set solver.
 */
TEST(sparse_quad_prog_solver, SimpleTestTwo)
{
  // // MATLAB CODE:
  // H = diag([1,3,7,5,1]);
  // zLow = [-1;  1; -3;  -8; 1];
  // zUpp = [ 2;  5; -2;  -2; 9];
  // Aeq = [ 1, 3, -3, -2, 1;
  //        -2, 3,  5,  1, 3];
  // beq = [34; 21];
  // f = [0;0;0;0;0]; A = []; b = [];
  // [z, objVal, exitCode] = quadprog(H,f,A,b,Aeq,beq,zLow,zUpp);
  // >> z = [1.8; 5.0; -2.0; -2.0; 7.2];  // solution decision variables
  // >> objVal = 89.04;  // solution objective function value

  int maxIter = 5;
  double tol = 1e-10;
  Eigen::VectorXd xLow(5); xLow << -1, 1, -3, -8, 1;
  Eigen::VectorXd xUpp(5); xUpp << 2, 5, -2, -2, 9;
  Eigen::VectorXd w(5); w << 1,3,7,5,1;
  Eigen::MatrixXd A(2, 5);
  A.row(0) << 1, 3, -3, -2, 1;
  A.row(1) << -2, 3,  5,  1, 3;
  Eigen::VectorXd b(2); b << 34, 21;

  // Solve:
  sns_ik::SparseQuadProgSolver solver;
  sns_ik::SparseQuadProgSolver::Result result = solver.solve(xLow, xUpp, w, A, b, maxIter, tol);
  tol = 1e-6;
  ASSERT_NEAR(result.objVal, 89.04, tol);
  ASSERT_NEAR(result.soln(0), 1.8, tol);
  ASSERT_NEAR(result.soln(1), 5.0, tol);
  ASSERT_NEAR(result.soln(2), -2.0, tol);
  ASSERT_NEAR(result.soln(3), -2.0, tol);
  ASSERT_NEAR(result.soln(4), 7.2, tol);
}

/*************************************************************************************************/

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
