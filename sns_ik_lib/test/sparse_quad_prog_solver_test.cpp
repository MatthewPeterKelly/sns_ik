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
  // Test problem:
  sns_ik::rng_util::setRngSeed(83907, 17302);
  int nVar = 2;
  int nCst = 1;
  int maxIter = 100;
  double tol = 1e-10;
  Eigen::VectorXd xLow = sns_ik::rng_util::getRngVectorXd(0, nVar, -1.0, 0.1);
  Eigen::VectorXd xUpp = sns_ik::rng_util::getRngVectorXd(0, nVar, 0.1, 1.0);
  Eigen::VectorXd w = sns_ik::rng_util::getRngVectorXd(0, nVar, 0.1, 1.0);
  Eigen::MatrixXd A = sns_ik::rng_util::getRngMatrixXd(0, nCst, nVar, -1.0, 1.0);
  Eigen::VectorXd b = sns_ik::rng_util::getRngVectorXd(0, nCst, -1.0, 1.0);

  A << 1.0, 1.0;
  b << 0.0;

  // Solve:
  sns_ik::SparseQuadProgSolver solver;
  sns_ik::SparseQuadProgSolver::Result result = solver.solve(xLow, xUpp, w, A, b, maxIter, tol);
  ASSERT_NEAR(result.objVal, 0.0, 1e-8);
}

/*************************************************************************************************/

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
