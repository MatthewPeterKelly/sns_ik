/** @file sparse_quad_prog_solver.hpp
 *
 *  @author: Matthew P. Kelly
 *  @brief: solve sparse quadratic program with simple objective and limit constraints
 *
 *  Solve the problem:
 *
 *  min:    0.5 * z' * H * z            (H must be diagonal)
 *
 *  subject to:   zLow <= z <= zUpp    (limit constraint)
 *                A * z = b            (linear equality constraint)
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

#include "sparse_quad_prog_solver.hpp"

#include <Eigen/SparseQR>
#include <limits>
#include <ros/console.h>

namespace sns_ik {

/**************************************************************************************************/

std::string getSparsityPatternString(const Eigen::SparseMatrix<double>& matrix)
{
  std::string str = "";
  Eigen::MatrixXd dense = Eigen::MatrixXd(matrix);
  size_t nRows = matrix.rows();
  size_t nCols = matrix.cols();
  for (size_t iRow = 0; iRow < nRows; iRow++) {
    str += "|";
    for (size_t iCol = 0; iCol < nCols; iCol++) {
      double value = dense(iRow,iCol);
      if (std::fabs(value) < 1e-10) {
        str += " ";
      } else {
        str += "X";
      }
    }
    str += "|\n";
  }
  return str;
}

/*************************************************************************************************/

SparseQuadProgSolver::Result SparseQuadProgSolver::solve(const Eigen::VectorXd& xLow, const Eigen::VectorXd& xUpp,
                                   const Eigen::VectorXd& w,
                                   const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                   int maxIter, double tol)
{
  // Input validation
  nVar_ = w.size();
  nCst_ = b.size();
  result_.info = SparseQuadProgSolver::ExitCode::BadInput; // assume bad input until checks pass
  result_.soln = Eigen::VectorXd::Zero(nVar_);
  result_.objVal = std::numeric_limits<double>::max();
  result_.feasible = false;
  if (w.size() != xLow.size()) { ROS_ERROR("Bad input!  w.size() != xLow.size()"); return result_; }
  if (w.size() != xUpp.size()) { ROS_ERROR("Bad input!  w.size() != xUpp.size()"); return result_; }
  if (w.size() != A.cols()) { ROS_ERROR("Bad input!  w.size() != A.cols()"); return result_; }
  if (b.size() != A.rows()) { ROS_ERROR("Bad input!  b.size() != A.rows()"); return result_; }
  result_.info = SparseQuadProgSolver::ExitCode::InternalError; // this will be overwritten


  // Verify that the limits are in the correct order:
  for (int iVar = 0; iVar < nVar_; iVar++) {
    if (xUpp(iVar) < xLow(iVar)) { ROS_ERROR("Bad input!  xUpp(iVar) < xLow(iVar)"); return result_; }
  }

  // Initialize the data structures:
  xLow_ = xLow; xUpp_ = xUpp;
  w_ = w; A_ = A; b_ = b;
  tol_ = tol;
  initializeTripletList();


  /* DEBUG */ ROS_INFO_STREAM("xLow_: " << xLow_.transpose());
  /* DEBUG */ ROS_INFO_STREAM("xUpp_: " << xUpp_.transpose());
  /* DEBUG */ ROS_INFO_STREAM("w_: " << w_.transpose());
  /* DEBUG */ ROS_INFO_STREAM("A_:\n" << A_);
  /* DEBUG */ ROS_INFO_STREAM("b_: " << b_.transpose());


  // Initialize the active set by assuming that no constraints are active:
  actSetLow_ = std::vector<bool>(nVar_, false);
  actSetUpp_ = std::vector<bool>(nVar_, false);

  // Main iteration loop
  for (int iter = 0; iter < maxIter; iter++) {
    /* DEBUG */ ROS_INFO("Iteration: %d", iter);

    // Solve the QP with the current active set:
    if (!solveWithActiveSet()) {
      ROS_ERROR("Failed to solve QP given current active set!");
      return result_;
    }

    // Update the active set:
    if (!updateActiveSet()) {
      ROS_ERROR("Failed to update the active set!");
      return result_;
    }

    // Check for convergence:
    if (result_.info == SparseQuadProgSolver::ExitCode::Success) {
      /* HACK */ ROS_INFO("Done!");
      return result_;  // successful convergence
    }
  }

  // Maximum iteration reached!
  if (result_.feasible) {
    result_.info = SparseQuadProgSolver::ExitCode::MaxIter;
  } else {
    result_.info = SparseQuadProgSolver::ExitCode::Infeasible;
  }
  return result_;
}

/*************************************************************************************************/

bool SparseQuadProgSolver::solveWithActiveSet()
{
  // Update the data structures that store the active set constraints
  removeActiveSetConstraints();  // remove the old constraints
  appendActiveSetConstraints();  // add new ones

  /* DEBUG */ ROS_INFO_STREAM("d_: " << d_.transpose());

  // Create the vector of constraints:
  int nDecVar = nVar_ + nCst_ + d_.size();
  Eigen::VectorXd cstVec(nDecVar);
  cstVec << Eigen::VectorXd::Zero(nVar_), b_, d_;

  // Create the sparse constraint matrix:
  Eigen::SparseMatrix<double> cstMat(nDecVar, nDecVar);
  cstMat.setFromTriplets(triplets_.begin(), triplets_.end());
  cstMat.makeCompressed();  // this is important for the QR decomposition to work properly

  /* DEBUG */ ROS_INFO("cstMat: \n%s", getSparsityPatternString(cstMat).c_str());

  // Solve the linear system
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver(cstMat);
  if(solver.info() != Eigen::ComputationInfo::Success) {
    ROS_ERROR("Failed to decompose the sparse linear system!");
    result_.info = SparseQuadProgSolver::ExitCode::InternalError;
    return false;
  }
  decVar_ = solver.solve(cstVec);
  return true;
}

/*************************************************************************************************/

bool SparseQuadProgSolver::updateActiveSet()
{
  /* DEBUG */ ROS_INFO_STREAM("Full decision variables: " << decVar_.transpose());

  // TODO: clean up this method

  // Compute the most negative lagrange multiplier for the active set
  double lagMin = 0.0;  // minimum value of the lagrange mutliplier
  int iAct = 0; // index into the current active set
  int iRem = -1; // index of the variable to be removed from the active set
  for (int iVar = 0; iVar < nVar_; iVar++) {
    if (actSetLow_[iVar] || actSetUpp_[iVar]) {
      double lag = decVar_[nVar_ + nCst_ + iAct];
      if (lag < lagMin) {
        lagMin = lag;
        iRem = iVar;
      }
      iAct++;
    }
  }

  // Compute the largest constraint violation to add to the active set:
  double cstMax = 0.0;  // maximum constraint violation
  int iAdd = -1; // index of constraint to add
  for (int iVar = 0; iVar < nVar_; iVar++) {
    bool low = decVar_(iVar) < xLow_(iVar) - tol_;
    bool upp = decVar_(iVar) > xUpp_(iVar) + tol_;
    if (low || upp) {
      double err = 0.0;
      if (low) { err = xLow_(iVar) - decVar_(iVar); }
      if (upp) { err = decVar_(iVar) - xUpp_(iVar); }
      if (err > cstMax) {
        iAdd = iVar;
        cstMax = err;
      }
    }
  }

  // // Sanity check:
  // if (iAdd == iRem) {
  //   ROS_ERROR("Something went terribly wrong!");
  //   result_.info = SparseQuadProgSolver::ExitCode::InternalError;
  //   return false;
  // }

  // Add one element to the active set, if allowed
  if (iAdd >= 0) {
    /* DEBUG */ ROS_INFO("Adding constraint %d to the active set.", iAdd);
    bool low = decVar_(iAdd) < xLow_(iAdd) - tol_;
    bool upp = decVar_(iAdd) > xUpp_(iAdd) + tol_;
    if (low) { actSetLow_[iAdd] = true; }
    if (upp) { actSetUpp_[iAdd] = true; }
  }

  // Remove one element from the active set, if allowed:
  if (iRem >= 0 && iAdd == -1) {  // TODO: check this...
    /* DEBUG */ ROS_INFO("Removing constraint %d from the active set.", iRem);
    actSetLow_[iRem] = false;
    actSetUpp_[iRem] = false;
  }

  // Update the result data if the solution is feasible and better than previous
  bool isFeasible = iAdd < 0;  // no constraints added to the set
  if (isFeasible) {
    double objVal = computeObjectiveFunction();
    /* DEBUG */ ROS_INFO("Feasible solution with objective value: %f", objVal);
    if (objVal < result_.objVal) {
      result_.soln = decVar_.head(nVar_);
      /* DEBUG */ ROS_INFO_STREAM("New best objective value!  DecVars: " << result_.soln.transpose());
      result_.objVal = objVal;
      result_.feasible = true;
    }
  }

  // Check if we are done:
  bool converged = iRem < 0 && isFeasible;
  if (converged) {
    /* HACK */ ROS_INFO("Success!");
    result_.info = SparseQuadProgSolver::ExitCode::Success;
  }
  return true;
}

/*************************************************************************************************/

double SparseQuadProgSolver::computeObjectiveFunction()
{
  double objVal = 0.0;
  for (int iVar = 0; iVar < nVar_; iVar++) {
    objVal += w_[iVar] * decVar_[iVar] * decVar_[iVar];
  }
  return 0.5 * objVal;
}

/*************************************************************************************************/

void SparseQuadProgSolver::initializeTripletList()
{
  // Memory allocation for the triplet vector
  nCore_ = 0;
  nCore_ += nVar_;  // diagonal elements in H matrix
  nCore_ += 2 * nVar_ * nCst_;  // elements in both blocks of the A matrix
  int nnz = nCore_ + nVar_;  // core elements + maximum number of active set constraints
  triplets_.clear();
  triplets_.reserve(nnz);

  // Set the data for the H matrix:
  for (int iVar = 0; iVar < nVar_; iVar++) {
    // elements of the H matrix (diagonal)
    triplets_.push_back(Eigen::Triplet<double>(iVar, iVar, w_(iVar)));
  }
  for (int iVar = 0; iVar < nVar_; iVar++) {
    // elements of the A matrix (needed in two places)
    for (int iCst = 0; iCst < nCst_; iCst++) {
      double a = A_(iCst, iVar);
      triplets_.push_back(Eigen::Triplet<double>(nVar_ + iCst, iVar, a));
      triplets_.push_back(Eigen::Triplet<double>(iVar, nVar_ + iCst, a));
    }
  }
  /* DEBUG */ ROS_INFO("Initial triplet list:");
  printTriplets();  // HACK
}

/*************************************************************************************************/

void SparseQuadProgSolver::removeActiveSetConstraints()
{
  int nSize = triplets_.size();  // total number of elements
  int nRemove = nSize - nCore_;  //numer of elements to be removed
  if (nRemove > 0) {
    triplets_.erase(triplets_.begin() + nCore_, triplets_.begin() + nSize);
  }
}

/*************************************************************************************************/

int SparseQuadProgSolver::countActiveConstraints()
{
  int nActive = 0;
  for (int iVar = 0; iVar < nVar_; iVar++) {
    bool low = actSetLow_[iVar];
    bool upp = actSetUpp_[iVar];
    if (low && upp) {
      ROS_ERROR("Cannot have both lower and upper limits active at once!");
      result_.info = SparseQuadProgSolver::ExitCode::InternalError;
      return -1;
    }
    if (low || upp) {
      nActive++;
    }
  }
  return nActive;
}

/*************************************************************************************************/

bool SparseQuadProgSolver::appendActiveSetConstraints()
{
  // Compute the number of active constraints
  int nActive = countActiveConstraints();
  if (nActive < 0) { return false; }
  d_.resize(nActive);
  /* DEBUG */ ROS_INFO("Active set size: %d", nActive);

  if (int(triplets_.size()) != nCore_) {
    ROS_ERROR("Cannot append active set constraint: triplet vector invalid size!");
    result_.info = SparseQuadProgSolver::ExitCode::InternalError;
    return false;
  }
  int iAct = 0;  // index of the current active set constraint
  for (int iVar = 0; iVar < nVar_; iVar++) {
    if (actSetLow_[iVar]) {
      d_(iAct) = -xLow_(iVar);
      triplets_.push_back(Eigen::Triplet<double>(nVar_ + nCst_ + iAct, iVar, -1.0));
      triplets_.push_back(Eigen::Triplet<double>(iVar, nVar_ + nCst_ + iAct, -1.0));
      iAct++;
    }
    if (actSetUpp_[iVar]) {
      d_(iAct) = xUpp_(iVar);
      triplets_.push_back(Eigen::Triplet<double>(nVar_ + nCst_ + iAct, iVar, 1.0));
      triplets_.push_back(Eigen::Triplet<double>(iVar, nVar_ + nCst_ + iAct, 1.0));
      iAct++;
    }
  }
  /* DEBUG */ ROS_INFO("Triplets with active set:");
  printTriplets();  // HACK
  return true;
}

/*************************************************************************************************/

void SparseQuadProgSolver::printTriplets()
{
  for (auto tri : triplets_) {
    ROS_INFO("Triplet(%d, %d) = %f", tri.row(), tri.col(), tri.value());
  }
}

/*************************************************************************************************/

}  // namespace sns_ik
