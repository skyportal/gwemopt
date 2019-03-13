"""
Use extended Hungarian algorithm to minimize/maximize the cost/probability of the scheduling matrix
"""

__author__ = "Javed Rana <javed@iucaa.in>"


### Imports
import numpy as np
import multiprocessing as mp

class Hungarian:
    '''Use Hungarian algorithm to minimize the cost'''

    def __init__(self, matrix, profit_matrix=False):
        '''Input the cost matrix.
        If the profit_matrix is true then make it the cost matrix
        '''
        self.assigned_points = []
        newmatrix = np.array(matrix)
        ### Padding of zeros if the matrix is not balanced.
        matrix_shape = newmatrix.shape
        max_size = max(matrix_shape)
        if max_size > 100*matrix_shape[1]:
            max_size = 100*matrix_shape[1]
            newmatrix = newmatrix[:max_size]
        matrix_shape = newmatrix.shape
        pad_row = max_size - matrix_shape[0]
        pad_column = max_size - matrix_shape[1]
        newmatrix = np.pad(newmatrix, ((0,pad_row),(0,pad_column)), 'constant', constant_values=(0.))
        
        if profit_matrix:
            max_element = newmatrix.max()
            matrix_shape = newmatrix.shape
            equal_matrix = np.ones(matrix_shape, dtype=int) * max_element
            newmatrix = equal_matrix - newmatrix
        self.cost_matrix = newmatrix.copy()
        self.shape = matrix_shape


    ### Phase 1
    ########################################################################
    ### Step 1
    def row_column_subtraction(self):
        """Subtract the minimum element of each row from that row"""
        if self.cost_matrix is not None:
            # Minimum element subtraction from each row 
            for index, row in enumerate(self.cost_matrix):
                self.cost_matrix[index] -= row.min()
            # Minimum element subctration from each column
            for index, column in enumerate(self.cost_matrix.T):
                self.cost_matrix.T[index] -= column.min()

    ### Phase 2
    ### Step 2
    def calculate(self):
        """Return the optimal solution"""
        ### Step 1
        self.row_column_subtraction()

        ### Step 2
        itr = 0
        optimal_row = []
        while len(optimal_row) < len(self.cost_matrix):
            ### Initial assigning of the jobs
            zerolines = ZeroLines(self.cost_matrix)
            zerolines.calculate()
            init_assignment = zerolines.assigned_points

            check_matrix = self.cost_matrix.copy()
            ### Mark all the zeros with minimum lines
            markedRC = markRowColumn(self.cost_matrix)
            markedRC.calculate(init_assignment)
            optimal_line = markedRC.optimal_line
            ### Check whether any farther improvement is possible
            if (check_matrix == self.cost_matrix).all():
                break
            if optimal_line == len(self.cost_matrix):
                break
        self.assigned_points = zerolines.assigned_points

### Step 2
### Find the minimum line to cover the all zeros
class ZeroLines:
    '''Cover the zeros assigning in the the jobs in rows and collumns
    If a row has one zero, assign that row.
    If a row has more than one zero skip that row.
    After all row assignment, assign the columns with one zero.
    If a column has more than one zeroskip that column.
    Repeat the row and column assignments untill all the zeros are eighter assigned or crossed
    '''

    def __init__(self, cost_matrix):
        """Input cost matrix and save the position of the optimal cost"""
        self.cost_matrix = cost_matrix
        self.optimal_row = []
        self.optimal_column = []
        self.zero_mask = [] 
        self.position_matrix = []
        self.assigned_points = []

    def rowassign(self, row_mask, matrix_column):
        '''Find the zeros in the row and return the column index of the first zero'''
        return matrix_column[row_mask][0]

    def newmatrix(self, delete_index, axis=0):
        '''Delete the assigned column'''
        self.zero_mask = np.delete(self.zero_mask, delete_index, axis=axis)
        self.position_matrix = np.delete(self.position_matrix, delete_index, axis=axis)

    def deleteindex(self, row_mask):
        '''Return the index to delete'''
        ar = np.arange(len(row_mask))
        return ar[row_mask][0]

    def rowScanning(self):
        '''If a row has one zero, assign that row.
        If a row has more than one zero skip that row.
        '''
        rowlen = len(self.zero_mask)
        for row_index in range(rowlen):
            row_mask = self.zero_mask[row_index]
            if row_mask.sum()==1:
                cindex_assignrow = self.rowassign(row_mask, self.position_matrix[row_index])
                self.assigned_points.append(cindex_assignrow)
                delete_index = self.deleteindex(row_mask)
                ### The changed matrix
                self.newmatrix(delete_index, axis=1)

    def columnScanning(self):
        '''If a column has one zero, assign that column.
        If a column has more than one zeroskip that column.
        '''
        columnlen = len(self.zero_mask.T)
        for column_index in range(columnlen):
            column_mask = self.zero_mask[:,column_index]
            if column_mask.sum()==1:
                rindex_assigncolumn = self.rowassign(column_mask, self.position_matrix[:,column_index])
                self.assigned_points.append(rindex_assigncolumn)
                delete_index = self.deleteindex(column_mask)
                ### The changed matrix
                self.newmatrix(delete_index, axis=0)

    def coverExtraZeros(self):
        '''If extra zeros are left after row and column scanning,
        cover them by arbitrary assignment.
        '''
        rowlen = len(self.zero_mask)
        for row_index in range(rowlen):
            row_mask = self.zero_mask[row_index]
            if row_mask.any():
                cindex_assignrow = self.rowassign(row_mask, self.position_matrix[row_index])
                self.assigned_points.append(cindex_assignrow)
                delete_index = self.deleteindex(row_mask)
                ### The changed matrix
                self.newmatrix(delete_index, axis=1)

    def calculate(self):
        """Find the index of the rows and columns having zeros
        and the indexes of the zreos"""
        rowlen = len(self.cost_matrix)
        columnlen = len(self.cost_matrix.T)
        row_indexes = np.arange(rowlen)
        column_indexes = np.arange(columnlen)
        
        self.position_matrix = np.zeros(shape=(rowlen,columnlen,2), dtype=int)
        for ri in range(rowlen):
            for ci in range(columnlen):
                self.position_matrix[ri,ci] = [ri,ci]
        self.zero_mask = self.cost_matrix == 0.
        copy_zero_mask = self.zero_mask.copy()
        itr = 0
        while self.zero_mask.any():
            self.rowScanning()
            if self.zero_mask is not None:
                self.columnScanning()
            itr += 1
            if itr>100:
                self.coverExtraZeros()


### Mark the unassigned rows and 
class markRowColumn:
    '''Mark all rows having no assignments.
    Mark all (unmarked) columns having zeros in newly marked rows.
    Mark all rows having assignments in newly marked columns.
    Repeat for all non-assigned rows.
    Draw lines through all marked columns and unmarked rows.
    '''

    def __init__(self, cost_matrix):
        '''Input cost matrix and the assigned row/columns'''
        self.cost_matrix = cost_matrix
        self.marked_rows = []
        self.unmarked_rows = []
        self.marked_columns = []
        self.unmarked_columns = []
        self.optimal_line = 0

    def zeroUnassignedRow(self, row):
        '''Find the zero in the unmarked rows'''
        row_mask = row == 0.
        if row_mask.any():
            ar = np.arange(len(row))
            return ar[row_mask]
        else: return []

    def uncoveredMatrix(self):
        '''Delete the all marked columns and unmarked rows'''
        f_matrix = np.delete(self.cost_matrix, self.marked_columns, axis=1)
        f_matrix = np.delete(f_matrix, self.unmarked_rows, axis=0)
        return f_matrix

    def __mark_row_column(self, unassigned_rows, assigned_columns, assigned_rows):
        '''Mark all (unmarked) columns having zeros in newly marked row.
        Mark all rows having assignments in newly marked columns.
        '''
        def _marking():
            '''First marking of the rows and columns from the unassigned 
            rows
            '''
            pool = mp.Pool(mp.cpu_count())
            cindex = pool.map(self.zeroUnassignedRow, [self.cost_matrix[ri] for ri in self.marked_rows])
            pool.close()
            for i in range(len(cindex)):
                # cindex = self.zeroUnassignedRow(self.cost_matrix[ri])
                for ci in cindex[i]:
                    self.marked_columns.append(ci)
            for ci in self.marked_columns:
                if ci in assigned_columns:
                    index = np.argmin(abs(assigned_columns-ci))
                    self.marked_rows.append(assigned_rows[index])
            self.marked_rows = list(set(self.marked_rows))
            self.marked_columns = list(set(self.marked_columns))
            self.unmarked_rows = np.delete(np.arange(len(self.cost_matrix)), self.marked_rows)

        r_matrix = self.uncoveredMatrix()
        r_matrix_mask = r_matrix == 0
        itr = 0
        while r_matrix_mask.any():
            _marking()
            r_matrix = self.uncoveredMatrix()
            r_matrix_mask = r_matrix == 0

    def minElementSub(self, f_matrix, intersect_points):
        '''Subtract the minimum element from all uncovered element
        Add the minimum element to all elements covered by two lines'''
        min_element = np.min(f_matrix)
        ### Subtract the minimum element from all uncovered element
        for ci in self.unmarked_columns:
            for ri in self.marked_rows:
                self.cost_matrix[ri,ci] -= min_element
        ### Add the minimum element to all elements covered by two lines
        for point in intersect_points:
            self.cost_matrix[point[0],point[1]] += min_element 

    def calculate(self, assigned_points):
        '''Calculate marked'''
        assigned_rows = (np.array(assigned_points).T)[0]
        assigned_columns = (np.array(assigned_points).T)[1]
        self.marked_rows = [ri for ri in range(len(self.cost_matrix)) if ri not in assigned_rows]
        unassigned_rows = self.cost_matrix[self.marked_rows]

        self.__mark_row_column(unassigned_rows, assigned_columns, assigned_rows)
        self.marked_columns = np.unique(np.array(self.marked_columns))
        self.marked_rows = np.unique(np.array(self.marked_rows))
        self.unmarked_rows = np.delete(np.arange(len(self.cost_matrix)), self.marked_rows)
        self.unmarked_columns = np.delete(np.arange(len(self.cost_matrix)), self.marked_columns)
        intersect_points = []
        for ri in self.unmarked_rows:
            for ci in self.marked_columns:
                intersect_points.append([ri,ci])

        f_matrix = self.uncoveredMatrix()
        if len(f_matrix):
            self.minElementSub(f_matrix, intersect_points)
        self.optimal_line = len(self.unmarked_rows) + len(self.marked_columns)

