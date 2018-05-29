/** @file sparse_quad_prog_solver.hpp
 *
 *  @author: Matthew P. Kelly
 *  @brief: solve sparse quadratic program with simple objective and limit constraints
 *
 *  Solve the problem:
 *
 *  min:    0.5 * x' * H * x            (H must be diagonal)
 *  subject to:   xLow <= x <= xUpp     (limit constraint)
 *                A * x = b             (linear equality constraint)
 *
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

#ifndef SNS_IK_SPARSE_QUAD_PROG_SOLVER_H_
#define SNS_IK_SPARSE_QUAD_PROG_SOLVER_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace sns_ik {

/**************************************************************************************************/

/*
 * This helper function is primarily intended for debugging. It returns a string where each
 * non-zero entry in the matrix is given "X" where as a zero element is given " ".
 */
std::string getSparsityPatternString(const Eigen::SparseMatrix<double>& matrix);

/**************************************************************************************************/

class SparseQuadProgSolver
{

public:

  /**
   * Enum to store solver exit code
   */
  enum struct ExitCode {
    Success,  // converged to the optimal solution
    MaxIter,  // failed to converge, but the solution is feasible
    Infeasible,  // no feasible solution was found
    BadInput,  // user input was bad
    InternalError  // something went wrong in the algorithm itself
  };


  /**
   * Struct to store the solution to the optimization problem
   */
  struct Result {
    SparseQuadProgSolver::ExitCode info;  // information about the return status
    Eigen::VectorXd soln;  // decision variables for the optimal solution
    double objVal;  // value of the objective function
    bool feasible;  // is the solution feasible?
  };

  /**
   * Default constuctor
   */
  SparseQuadProgSolver() {};

  /**
   * Solve the QP:
   *  min:    0.5 * x' * H * x            (H must be diagonal)
   *  subject to:   xLow <= x <= xUpp     (limit constraint)
   *                A * x = b             (linear equality constraint)
   *
   * @param xLow: lower bound on decision variables, length = nVar
   * @param xUpp: upper bound on decision variables, length = nVar
   * @param w: weight on each decision variable quadratic cost, length = nVar
   * @param A: linear constraint matrix, size = [nCst, nVar]
   * @param b: linear constraint vector, length = nCst
   * @param maxIter: maximum number of iterations
   * @param tol: constraint tolerance
   * @return: result (includes solution, exit code, objval, and the active set of limits)
   */
  Result solve(const Eigen::VectorXd& xLow, const Eigen::VectorXd& xUpp, const Eigen::VectorXd& w,
               const Eigen::MatrixXd& A, const Eigen::VectorXd& b, int maxIter, double tol);

private:

  // TODO: clean up documentation in this section

  /**
   * Solve the QP given the current active set
   * After calling this method the decVars_ member variable will containt the solution
   * @return: true iff successful
   */
  bool solveWithActiveSet();

  /**
   * Update the active set. If active set is valid, then set result_.info to Success.
   * This method modifies the actSetLow_ and actSetUpp_ vectors of boolean values
   * @return: true iff successful
   */
  bool updateActiveSet();

  /**
   * This method initializes the vector of triplets, added data for the H and A matricies, and
   * allocation memory for the active constraints.
   */
  void initializeTripletList();

  /**
   * Remove the triplets associated with the active set constraints
   * After this operation triplets_.size() == nVar_ * (1 + 2 * nCst_)
   */
  void removeActiveSetConstraints();

  /**
  * @return: number of active constraints (or -1 if error)
  */
  int countActiveConstraints();

  /**
   * Append the active set of constraints to the triplet list.
   * - count the number of currently active constraints
   * - add constraints to both the triplet list and the "d_" vector of constraint constants
   * @return: true iff successful (false if the triplet list has any active constraints present)
   */
  bool appendActiveSetConstraints();

  /**
   * Compute the value of the objective function for a candidate set of decision variables
   * @return: 0.5 * x' * H * x    (H = diag(w))
   */
  double computeObjectiveFunction();

  /**
   * Diagnostics method: print the current set of triplets to ROS_INFO()
   */
  void printTriplets();

  // Local copy of the result
  SparseQuadProgSolver::Result result_;

  // Local copy of the problem size
  int nVar_;
  int nCst_;

  // Local copy of problem data, see documentation for solve()
  Eigen::VectorXd xLow_;
  Eigen::VectorXd xUpp_;
  Eigen::VectorXd w_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;

  // Which decision variables are in the active set?
  std::vector<bool> actSetLow_; // lower bound active set
  std::vector<bool> actSetUpp_; // upper bound active set

  /**
   * Triplet representation of the sparse linear system.
   * The first set of triplets, corresponding to the upper left block of (nVar + nCst) is not
   * modified after initial construction. The active set constraints are appended or deleted from
   * the end of the triplet list on each iteration.
   */
  std::vector<Eigen::Triplet<double>> triplets_;

  // Number of "core" non-zero elements in the QP = nVar_ * (1 + 2 * nCst_)
  int nCore_;

  // values of the active set constraints
  Eigen::VectorXd d_;

  // current value of the decision variables  (may or may not be the best so far)
  Eigen::VectorXd decVar_;

  // tolerance on the constraint checks
  double tol_;

};

/**************************************************************************************************/

}  // namespace sns_ik

#endif  // SNS_IK_SPARSE_QUAD_PROG_SOLVER_H_
