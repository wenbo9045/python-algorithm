
def add(matrix_a, matrix_b):
    rows = len(matrix_a)
    columns = len(matrix_a[0])
    matrix_c = []
    for i in range(rows):
        list_1 = []
        for j in range(columns):
            val = matrix_a[i][j] + matrix_b[i][j]
            list_1.append(val)
        matrix_c.append(list_1)
    return matrix_c

def scalarMultiply(matrix , n):
    return [[x * n for x in row] for row in matrix]

def multiply(matrix_a, matrix_b):
    matrix_c = []
    n = len(matrix_a)
    for i in range(n):
        list_1 = []
        for j in range(n):
            val = 0
            for k in range(n):
                val = val + matrix_a[i][k] * matrix_b[k][j]
            list_1.append(val)
        matrix_c.append(list_1)
    return matrix_c

def identity(n):
    return [[int(row == column) for column in range(n)] for row in range(n)] 

def transpose(matrix):
    return map(list , zip(*matrix))

def minor(matrix, row, column):
    minor = matrix[:row] + matrix[row + 1:]
    minor = [row[:column] + row[column + 1:] for row in minor]
    return minor

def determinant(matrix):
    if len(matrix) == 1: return matrix[0][0]
    
    res = 0
    for x in range(len(matrix)):
        res += matrix[0][x] * determinant(minor(matrix , 0 , x)) * (-1) ** x
    return res

def inverse(matrix):
    det = determinant(matrix)
    if det == 0: return None

    matrixMinor = [[] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrixMinor[i].append(determinant(minor(matrix , i , j)))
    
    cofactors = [[x * (-1) ** (row + col) for col, x in enumerate(matrixMinor[row])] for row in range(len(matrix))]
    adjugate = transpose(cofactors)
    return scalarMultiply(adjugate , 1/det)

def search_in_a_sorted_matrix(mat, m, n, key):
    i, j = m - 1, 0
    while i >= 0 and j < n:
        if key == mat[i][j]:
            print('Key %s found at row: %s column: %s' % (key, i + 1, j + 1))
            return
        if key < mat[i][j]:
            i -= 1
        else:
            j += 1
    print('Key %s not found' % (key))

def main():
    matrix_a = [[12, 10], [3, 9]]
    matrix_b = [[3, 4], [7, 4]]
    matrix_c = [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]]
    matrix_d = [[3, 0, 2], [2, 0, -2], [0, 1, 1]]

    print('matrix add operation')
    print(add(matrix_a, matrix_b))
    print('matrix multiply operation')
    print(multiply(matrix_a, matrix_b))
    print('identity matrix')
    print(identity(5))
    print('minor matrix')
    print(minor(matrix_c , 1 , 2))
    print('determinant matrix')
    print(determinant(matrix_b))
    print('inverse matrix')
    print(inverse(matrix_d))

    mat = [
        [2, 5, 7],
        [4, 8, 13],
        [9, 11, 15],
        [12, 17, 20]
    ]
    x = int(input("Enter the element to be searched:"))
    print(mat)
    search_in_a_sorted_matrix(mat, len(mat), len(mat[0]), x)

if __name__ == '__main__':
    main()
