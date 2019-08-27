# -*- coding: utf-8 -*-
#   Author: Alan Bartlett, alanmb@math.washington.edu
#
#   Tools for 1d (and later 2d) substitutions
#
#   
#   To Add:
#       Visualize via substitution graphs: color cubes.
#       Visualize via turtle graphics: rotate and move.

#   INSTR = ([1,0,3,1],[3,0,2,1],[2,1,0,3],[2,0,0,1],[1,2,3,0],[2,3,0,3],[1,2,3,1],[0,3,0,1],[3,2,3,0],[3,2,1,1],[1,1,2,3],[0,3,1,2],[1,2,1,0],[3,1,0,1],[0,2,1,3],[2,1,3,0],[0,2,1,2],[0,2,1,2],[0,0,2,1],[1,3,0,3],[1,2,1,0],[0,3,0,1],[1,3,2,0],[2,1,0,1],[1,3,1,2],[0,1,3,2],[1,2,0,3],[3,2,1,1],[3,1,1,2],[3,0,1,2])

from numpy import absolute, floor_divide, dot, floor, kron, log, mod, nonzero, prod
from numpy import array, ndarray, newaxis, concatenate as array_smash, reshape
from numpy.linalg import eig, inv, matrix_power
from numpy.lib.twodim_base import diag
from itertools import product as cartesian, combinations
from fractions import gcd
from sympy import Matrix, Symbol, eye, zeros, sqrt
from sympy.solvers import solve


###  Various functions used 

# Turns a tuple of s-elements from the set [0,s) into a sxs matrix representing the map list[a] = b
instruction_matrix = lambda instr: Matrix(array([[i==instr[j] for j in range(len(instr))] for i in range(len(instr))],dtype=int));


# Lambda Functions

qadic = lambda n,q,k: tuple([tuple(mod(floor_divide(array(k),array(q)**i),array(q))) for i in range(n)]); # express q-adic expansion of k up to height n

instruction_matrix = lambda instr: array([[i==instr[j] for j in range(len(instr))] for i in range(len(instr))],dtype=int); # construct matrix with 1's in positions (a,b) if instr(a) = b

def location_set(q):
    # location_set is an iterative which enumerates the elements of the rectangle [0,q) (in Z^d)
    # the map is essentially just q-adic representation of the numbers [0,prod(q))
    if type(q) == int:
        q = (q,);
    #return [tuple(reversed(k)) for k in cartesian(*[range(qj) for qj in reversed(q)])];
    return [tuple(reversed(k)) for k in cartesian(*[range(qj) for qj in reversed(q)])];

def array_location_set(k):
    # returns an iterative which enumerates the indices of a matrix from the bottom left corner, reading left->right, bottom->top, consistent with R^n enumeration
    # the 2nd to last coordinate needs to the x-axis, the last coordinate is the y-axis but counts backwards from num_rows   
    if type(k) == int:
        return [(j,) for j in range(k)];
    elif len(k) == 1:
        return [(j,) for j in range(k[0])];
    else:
        return [tuple(reversed((j[-1],)+(-j[-2]-1,)+j[:-2])) for j in cartesian(*[range(kj) for kj in reversed(k)])];


class PRF:
    # PRF stands for primitive reduced form.  Its input is the substitution matrix
    # sqrt(float((X[k]*X[k].conjugate()).evalf()))
    def __init__(self,SUBST_MATRIX):
        # We compute the eigenvalues of substitution matrix, for 2 reasons:
        #   1. The multiplicity of Q (the spectral radius) gives the number of ergodic classes
        #       This is because the primitive components have primitive substitution matrices
        #   2  The eigenvalues of modulus Q give information on the indexes of primitivity; find the various h_i and multiply
        SUBST_MATRIX = Matrix(SUBST_MATRIX); # force type        
        e_vals = SUBST_MATRIX.eigenvals();
        modulus_e_vals = [sqrt(lam*lam.conjugate()).evalf() for lam in e_vals];
        Q = max(modulus_e_vals);
                
    
class INSTRUCTION:
    def __init__(self,instr):
        # an instruction represents a map from an alphabet to itself.
        # we represent the alphabet and instruction in a single list: 
        #   alphabet = range(s) where s = len(instr)
        #   instr[j] = the value the instruction takes on the j-th letter
        self.instr = instr;
        
        # a matrix representing the instruction, r_ij = 1 iff i = R(j)
        self.matrix;
        
    

        
class SUBSTITUTION:
    def __init__(self,q =(4,2,3),instructions=[[0,1,2],[2,0,1],[1,2,0],[0,2,1],[2,1,0],[1,0,2],[1,0,0],[0,1,0],[0,0,1],[2,1,1],[1,2,1],[1,1,2],[0,2,2],[2,0,2],[2,2,0],[1,0,0],[0,1,0],[0,0,1],[2,1,1],[1,2,1],[1,1,2],[0,2,2],[2,0,2],[2,2,0]]):
        # q is a tuple indicating the geometric inflation in R^d
        # instructions is a list of maps from A -> A, where A is the alphabet range(s) of s symbols 0, 1, ..., s-1
 
        if type(q) == int:
            self.q = (q,);                                  # The tuple describing the (d-dimensional) geometric inflation factor of the substitution
        else:
            self.q = q;                              
        self.Q = prod(self.q);                              # The numerical inflation rate
        self.d = len(self.q);                               # The dimension of the substitution
        self.s = len(instructions[0]);                      # The number of letters in the alphabet, represented by range(s)
        
        # Create Configuration: these are the instructions of the susbtitution, or how it replaces letters by position.
        # an instruction is a list of length s, where instruction[gamma] = alpha if that instruction maps gamma to alpha
        locset = self.location_set(self.q);
        self.configuration_array = ndarray(tuple(reversed(self.q)),list); 
        for k in range(len(locset)): self.configuration_array[tuple(reversed(locset[k]))]=instructions[k];
        
        # Create Substitution map: this is an nd-array where sub[gamma] = substitution_array for substitution on gamma
        self.sub = [gamma for gamma in range(self.s)];
        for gamma in range(self.s):
            self.sub[gamma] = ndarray(tuple(reversed(self.q)),int)
            for idx in zip(array_location_set(self.q),location_set(self.q)):
                self.sub[gamma][idx[0]] = self.configuration(idx[1])[gamma] 
        
        # Create instruction matrices:
        self.instructions = {LOC: INSTR for LOC,INSTR in zip(location_set(self.q),instructions)};        # This is the configuration
        self.instr_m = {LOC: instruction_matrix(self.instructions[LOC]) for LOC in location_set(self.q)};        

    def location_set(self,D):
        # location_set is an iterative which enumerates the elements of the rectangle [0,q) (in Z^d)
        # the map is essentially just q-adic representation of the numbers [0,prod(q))
        if type(D) == int:
            D = (D,);
        #return [tuple(reversed(k)) for k in cartesian(*[range(qj) for qj in reversed(q)])];
        return [tuple(reversed(k)) for k in cartesian(*[range(qj) for qj in reversed(D)])];
      
    def configuration(self,k):
        if type(k)==int:
            k = (k,)
        return self.configuration_array[tuple(reversed(k))];
    # Can now call with self.configuration[k][gamma] where k is a tuple from location_set, this will tell you what the substition replaces gamma with in the k-th position
    
    def GEN_INSTR(self,n,k):
        if type(k)==int:
            k = (k,)
        Rnk = array(eye(self.s));
        for k_i in qadic(n,self.q,k):
            Rnk = dot(Rnk,self.instr_m[k_i])
        return Rnk;
    
    def evaluate(self,n,k,gamma):
        # Compute S^n(gamma)_k
        Rnk = Matrix(self.GEN_INSTR(n,k));
        return list(Rnk[:,0]).index(1)
        
    # create an inflatable substitution map.
    def inflate(self,n,gamma):
        # Compute S^n(gamma)
        sub_word = ndarray((1,)*self.d,int);
        for k in range(n):
            word = sub_word;
            sub_word = [];
            for alpha in word:
                sub_word+=self.sub[alpha]
        
        return sub_word        
    
    def inflate_1d(self,n,gamma):
        # Compute S^n(gamma)
        sub_word = [gamma];
        for k in range(n):
            word = sub_word;
            sub_word = [];
            for alpha in word:
                sub_word+=list(self.sub[alpha])
        
        return sub_word
        
    def inflate_2d(self,n,gamma):
        # Compute S^n(gamma), by concatenating along each row (adding columns) then along columns (concatenate rows)
        n
    
    
        

        
    


class SUBS:
    def __init__(self):
        print('Importing Thue_Morse, Rudin_Shapiro, and Queffelec Substitutions ...')   
        self.Thue_Morse = SUBSTITUTION(2,[[0,1],[1,0]]);
        self.Rudin_Shapiro = SUBSTITUTION(2,[[0,3,0,3],[2,2,1,1]]);
        self.Rudin_Shapiro_2 = SUBSTITUTION(2,[[0,0,2,2],[3,1,1,3]]);
        self.Rudin_Shapiro_3 = SUBSTITUTION(2,[[0,0,2,2],[1,3,3,1]]);
        self.Queffelec = SUBSTITUTION(3,[[0,1,2],[0,2,1],[1,2,0]]);
        self.Queffelec_2 = SUBSTITUTION(3,[[0,1,2],[0,2,1],[1,0,2]]);
        
        
        print('Importing Table, a bijective substitution in the plane...')   
        #Table = SUBSTITUTION([2,2],[[1,0,2,3],[0,2,1,3],[3,1,2,0],[0,1,3,2]]); SUBS.append(Table);
        self.TABLE = SUBSTITUTION([2,2],[[1,0,2,3],[0,2,1,3],[3,1,2,0],[0,1,3,2]]);
        
        
        print('Importing N1S, a pure discrete substitution ...')
        self.N1S = SUBSTITUTION(3,[[2,1,0],[1,2,2],[2,2,1]]);
        
        print('Importing N2S, a nonbijective sub with two spectral classes ...')
        self.N2S = SUBSTITUTION(3,[[0,1,2,3,4],[0,1,2,4,3],[1,2,3,2,0]]); # -1 < v1 < 1    
         
        print('Importing H2B, a bijective substitution with height 2')
        self.H2B = SUBSTITUTION(3,[[0,1,2,3],[1,2,3,0],[2,3,0,1]]);
        
        print('Importing H3B, a bijective substitution with height 3')
        self.H3B = SUBSTITUTION(4,[[0,1,2,3,4,5],[1,2,3,4,5,0],[2,3,4,5,0,1],[3,4,5,0,1,2]]);
        
        print('Importing HPD, a substitution of nontrivial height, pure discrete')
        self.HPD = SUBSTITUTION(3,[[0,1,2],[1,0,0],[0,2,1]]);
        
        print('Importing bHPD, the pure base of HPD')
        self.bHPD = SUBSTITUTION(3,[[0,0],[0,1],[1,0]]);
        
        print('SUBS which Cause Errors:' )    
        self.periodic = SUBSTITUTION(2,[[0,1],[1,1]]);    
        
           
        print('Importing LCs, examples for Lax Chan.')
        self.LC = SUBSTITUTION(2, [[0,2,1,3],[1,0,3,2]]);
        self.LC2 = SUBSTITUTION(4,[[0,0,3,3],[1,1,2,2],[3,0,3,0],[1,2,1,2]]);
        self.LC3 = SUBSTITUTION(3,[[0,0,0,7,8,8,7,7,8],[1,3,5,5,3,1,1,3,5],[2,4,6,2,2,6,4,6,4]]);
        
        print('Importing Cyclic RS')
        self.CM = SUBSTITUTION(4,[[0,0,2,2],[1,1,3,3],[1,3,3,1],[2,0,0,2]]);
        self.N = SUBSTITUTION(4,[[0,0,2,2],[3,3,1,1],[0,2,2,0],[1,3,3,1]]);
        
        print('Importing B4S, a bijective substitution with four spectral classes ...')
        self.B4S = SUBSTITUTION(3,[[0,1,2,3,4,5],[1,2,3,4,5,0],[2,3,4,5,0,1]]);
            
        print('Importing RS2, a substitution in plane with Lebesgue component ...')
        # RS2 = SUBSTITUTION([2,2],[[0,0,0,4,4,4,4,0],[1,1,5,1,5,5,1,5],[2,6,2,2,6,2,6,6],[7,3,3,3,3,7,7,7]]); SUBS.append(RS2);
        self.rs2 = SUBSTITUTION(4,[[0,0,0,4,4,4,4,0],[1,1,5,1,5,5,1,5],[2,6,2,2,6,2,6,6],[7,3,3,3,3,7,7,7]]);
        
        
        print('Importing NF3, a 3 fold natalie type substitution')
        self.NF3 = SUBSTITUTION(3,[[0,1,2,1,2,0,2,0,1],[5,3,4,3,4,5,4,5,3],[7,8,6,8,6,7,6,7,8]]); 
        
        print('Importing NonPrim, a nonprimitive substitution with 2 ergodic classes')
        self.NonPrim = SUBSTITUTION(2,[[1,0,4,3,2,1],[4,2,1,2,0,4]]); 