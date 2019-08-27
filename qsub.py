#   Author: Alan Bartlett, alanmb@math.washington.edu
#
#   To do:
#
#       Add methods to SUBSTITUTION class allowing more interation with substitutions (user friendly input and output)
#       
#       Replace "Primitive check" with an "iterate flag" to prevent recursive computation with BISUB.
#           Make sure we aren't computing more stuff of BISUB than we need, for speed.
#           Make spectral components compute the substitution, we also don't want spectral data for these
#
#       Spectral algorithms may not work correctly for substitutions which are not primitive.  Currently several methods of SUBSTITUTION do not behave well for nonprimitive subs (like lambda_hat, for example)
#           For nonprimitive, return the primitive components and the transient part of a substitution.  At the moment, we need a new object to handle this.  
#
#       Need to make it optional to compute ExtremeK.  The eigenvalue solve is the most computationally intensive aspect of the software, by several orders of magnitude, especially for large substitutions.       
#           Create a lookup table based: spectral_classes, transient_class : ExtremeK
#           If value exists, use that.  Otherwise, compute them.
#
#       Make iSIGMA, an iterative SIGMA for faster plotting
#           plot lambda_j using plot(range(N),[self.lambda_hat(j,i) for i in range(N)])


from __future__ import print_function
from numpy import absolute, array, floor_divide, dot, floor, log, mod, kron, nonzero, reshape, pi, exp, power
from numpy.linalg import eig, inv, matrix_power
from numpy.lib.twodim_base import diag
from numpy.random import rand
from matplotlib import pyplot #, mpl
from itertools import product as cartesian, izip, combinations
from fractions import gcd
from sympy import Matrix, Symbol, eye, zeros
from sympy.solvers import solve
import turtle


###########################################################################
#
#   The following are a few tools for: 
#               q-adic arithmetic 
#               adic enumeration and indexing in Z^d
#               representationg maps on finite sets
#
###########################################################################
def lcm(*numbers):
    return reduce(lambda x,y: (x*y)/gcd(x,y), numbers,1)

def count_sort(x,y):
    if x.count(1)>y.count(1):
        return 1
    elif x.count(1)==y.count(1):
        return 0
    else:
        return -1

def matrix_sort(A,B):
    mat_sum = lambda M: (M*Matrix([1]*len(M)))[0]
    if mat_sum(A) > mat_sum(B):
        return -1
    elif mat_sum(A) == mat_sum(B):
        return 0
    else:
        return 1
        
def vec_less(v,w):
    return array([ v[i] <= w[i] for i in range(0,len(w))]).all()

# q should be an array, representing a Z^d integer.

# The following function computes the n-th order q-adic expansion of k
# The output is a list of length n, [k_0,k_1, ... , k_(n-1)]
#   and k_i is the i-th q-adic digit of k.

qadic = lambda n,q,k: tuple([tuple(mod(floor_divide(array(k),array(q)**i),array(q))) for i in range(n)]);

qsum = lambda q,K: tuple((array(K)*array([array(q)**i for i in range(len(K))])).sum(0))

carry = lambda p,q,j,k: floor_divide(mod(j,array(q)**p) +  mod(k, array(q)**p),array(q)**p)

matrix_index = lambda s,alpha_beta: tuple([mod(floor_divide(alpha_beta,s**i),s) for i in [1,0]]); #  alpha_beta ---> (alpha,beta) # s is the size of the alphabet.  alpha_beta is the linear index of the letter in the bialphabet. 

e_alpha = lambda s,alpha: eye(s)[alpha,:]; # Returns the representation of alpha in the vector space C^A.  sends alpha ---> e_alpha, the standard basis vector which is one at index alpha (alpha is 0,...,s-1)

E_alphabeta = lambda s,alpha_beta: e_alpha(s,matrix_index(s,alpha_beta)[0]).transpose()*e_alpha(s,matrix_index(s,alpha_beta)[1]);

orbit = lambda n,S: array(matrix_power(array(S,dtype='float64'),n) !=0, dtype=int);

def location_set(q):
    if type(q) == int:
        return zip(range(q));
    else:
        return zip(*reversed(tuple(zip(*cartesian(*(range(qj) for qj in reversed(q)))))));

prod_alphabet = lambda s,z: zip(*list(zip(*cartesian(*(range(sj) for sj in [s,z])))));

# Inverse to prod_alphabet takes ab and returns (a,b)
alpha_adic = lambda z,ab: (floor_divide(ab,z), mod(ab,z));

instruction_matrix = lambda instr: array([[i==instr[j] for j in range(len(instr))] for i in range(len(instr))],dtype=int);

# Kronecker Product of instructions, as a list.  
#   These give the instructions of a substitution product.  See SUBSTITUTION_PRODUCT
instr_kron = lambda instr1, instr2: [prod_alphabet(len(instr1),len(instr2)).index((instr1[idx[0]],instr2[idx[1]])) for idx in prod_alphabet(len(instr1),len(instr2))];

# Composition of instructions, as a list. Requries same size lists
#   These give the instructions of an iterated product.  See ITERATE
instr_comp = lambda instr1, instr2: [instr1[instr2[i]] for i in range(len(instr2))];

    
    
def diagonalize(A_list):
    # Attempts to simultaneously diagonalize the matrices in A_list.
    
    # Check that the matrices in A_list commute.      
    
    A = A_list[0]; # Replace by iterating over A_list.
    # Returns the matrix of eigenvectors of A, diagonalizing A
    spectrum = A.eigenvects();
    U = [];
    for eigenspace in spectrum:
        U = U + [eigenvector.transpose() for eigenvector in eigenspace[2]];
    return Matrix(U).transpose();

# A substitution is a collection of maps A --> A indexed on a location set, these maps are instructions.
#   s is the cardinality of A

# In the below, an instruction R is represented by the vector r in C^s with r_j = R(j)
# The instruction_matrix, however, is the representation of this function as a matrix operating on C^A.  (R)_ij = 1 if R(i)=j and 0 otherwise.

class SUBSTITUTION:
    def __init__(self,q,INSTRUCTIONS,iterate=1):
        # Create the substitution dictionary on the location set for q
        # Using the instructions from the list INSTRUCTIONS
        # iterate is a flag used to prevent recursive computation
        if type(q)==int:
            self.q = (q,)
            self.dimension = 1            
        else:
            self.q = q
            self.dimension = len(q)
        self.Q = array(self.q).prod();
        self.s = len(INSTRUCTIONS[0]);
    
        self.instructions = {LOC: INSTR for LOC,INSTR in izip(self.loc(),INSTRUCTIONS)};        # This is the configuration
        self.instr_m = {LOC: instruction_matrix(self.instructions[LOC]) for LOC in self.loc()};        
        


        self.sub = [list(row) for row in array([ self.instructions[j] for j in self.loc()],dtype=int).transpose()];
        self.substitution_matrix = array([self.instr_m[i] for i in self.loc()]).sum(0);
#        self.coinc_m = array([kron(self.instr_m[i],self.instr_m[i]) for i in self.loc()]).sum(0);

        # If primitive, 
        if (matrix_power(self.substitution_matrix,self.s) == 0).all(): # is not primitive.  Reduce to primitive components, then return the primitive components.  Also, package the transient blocks in some way as well.  
            print("Matrix may not be Primitive.  Running software anyway.")
        
        # If iterate, compute bisubstitution and gather spectral data.  This is done so that when computing the bisubstitution, we do not iterate and compute the bisubstitution of the bisubstitution.
        if iterate:
            self.BISUB = SUBSTITUTION_PRODUCT(self,self,0);
            #self.ergodic_classes, self.transient_class = ERG_CLASS(self.BISUB);
            self.spectral_classes, self.transient_class = ERG_CLASS(self.BISUB);
            self.spectral_classes = list(self.spectral_classes); self.transient_class = list(self.transient_class);
            self.spectral_classes.sort(); # The first ergodic class should now be the diagonal pairs.
            
            # We need to pair the antisymmetric ergodic classes as the matrix of v must be Hermitian Positive Definite

            
            # Use the spectral classes to generate the matrix V for the vector v in K.
            T = tuple(self.transient_class);
            self.v = [Symbol('v'+str(i)) for i in range(0,len(self.spectral_classes))];  erg_idx = 0;# generate k-1 symbols v1,...,v_{k-1} and initialize a count index
            self.v[0] = 1;
            self.V = Matrix(zeros(self.s));  # create matrix self.V: first fill in on primary spectral class
            self.trans_c = Matrix([0]*len(T)).transpose();   # Store the transient pairs
            
            self.spectral_matrix = [];
            self.spectral_matrix = [Matrix(array([E_alphabeta(self.s,alphabeta).tolist() for alphabeta in E]).sum(0)) for E in self.spectral_classes];
            if self.transient_class:
                #self.transient_matrix = Matrix(array([E_alphabeta(self.s,alphabeta).tolist() for alphabeta in self.transient_class]).sum(0));
                # USE FORMULA
                #
                #   P_T v = P_T ( QI - C_S^t P_T)^(-1) P_T C_S^t v
            
                #   P_tran = PT ( Q I - CSt PT).inv() PT CSt 
                #
                PT = Matrix(array([(e_alpha(self.s**2,alphabeta).transpose()*e_alpha(self.s**2,alphabeta)).tolist() for alphabeta in self.transient_class]).sum(0));
                CSt = Matrix(self.BISUB.substitution_matrix).transpose();
                P_tran = PT*(self.Q*eye(self.s**2) - CSt*PT).inv() * PT * CSt;
                self.transient_matrix = [(P_tran*E.reshape(1,self.s**2).transpose()).reshape(self.s,self.s) for E in self.spectral_matrix];                
                
            
            
            for E in self.spectral_classes:
                self.V = self.V + self.v[erg_idx]*self.spectral_matrix[erg_idx];
                if T:
                    self.trans_c = self.trans_c + self.v[erg_idx] * Matrix([1]*len(E)).transpose() * Matrix(self.BISUB.substitution_matrix).extract(E,T);
                erg_idx = erg_idx + 1;
            # vK is solved for using the eigenvector condition in block coordinates
            self.vK = self.trans_c * (self.Q * Matrix(eye(len(T))) - Matrix(self.BISUB.substitution_matrix).extract(T,T)).inv()       
            for t in range(len(T)):
                self.V[matrix_index(self.s,T[t])] = self.vK[t]    # Fill in self.V with transient pairs
            
        # BEGIN CORRELATION VECTOR - this is the portion that depends on the configuration
            # Compute Perron Eigenvector of sub_m
            E = eig(self.substitution_matrix); IDX = list(E[0]).index(E[0].max()); E = [row[IDX] for row in E[1]]; 
            self.perron = E/array(E).sum(0);
    
            # Initialize SIG over unit_cube
            unit_cube = [c for c in location_set([2]*self.dimension)]; unit_cube.sort(count_sort)
            self.SIG = {unit_cube.pop(0):reshape(diag(self.perron),[self.s**2,1])};
            
            # This requires us to solve using a recursive identity
            while unit_cube:
                c = unit_cube.pop(0);
                Ac = array(0*eye(self.s**2)); B = reshape(array([0]*(self.s**2)),[self.s**2,1]); 
                loc_set = self.loc()
                for loc in loc_set:
                    S = kron(self.instr_m[tuple(mod(array(loc)+array(c),self.q))],self.instr_m[loc]);
                    if tuple(carry(1,self.q,loc,c)) == c:
                        # Then this term contributes to SIG(c)
                        Ac = Ac + S
                    else:
                        B = B + dot(S,self.SIG[tuple(carry(1,self.q,loc,c))])
                self.SIG[c] = dot(inv(self.Q * eye(self.s**2) - Ac),B)
   
        else: # Substitution Imprimitive
            # print 'Matrix Nonprimitive:', self.instructions
            self.BISUB = None
            self.spectral_classes = None
            self.transient_class = None
            self.spectral_hull = None

    def loc(self):
        return location_set(self.q);
    
    
    # BEGIN TEST
    
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
        
    def inflate(self,n,gamma):
        # Compute S^n(gamma)
        sub_word = [gamma];
        for k in range(n):
            word = sub_word;
            sub_word = [];
            for alpha in word:
                sub_word+=list(self.sub[alpha])
        
        return sub_word
        
        
    # END TEST
                    
    def SIGMA(self,k):
        if type(k)==int:
            k = (k,)
        if (vec_less((0,)*self.dimension, k) and vec_less(k, (1,)*self.dimension)):
            return Matrix(self.SIG[k])
        else:
            p = int(max(floor(1+log(absolute(k))/log(self.q))))+1; # add 1 to ensure no errors, can always compute to a higher index.
            # compute p-th instructions
            pINSTR = {};
            for loc in location_set(tuple(array(self.q)**p)):
                R = array(eye(self.s))
                for j in qadic(p,self.q,loc):
                    R = dot(R,self.instr_m[j]);
                pINSTR[loc] = R
            SIG = reshape(array([0]*(self.s**2)),[self.s**2,1])
            for loc in location_set(array(self.q)**p):
                SIG = SIG + dot(kron(pINSTR[tuple(mod(array(loc)+array(k),array(self.q)**p))],pINSTR[loc]),self.SIG[tuple(carry(p,self.q,loc,k))]);
            return Matrix(SIG / self.Q**p)

        
    def spectral_hull(self,spectrum=False):
    # Computes the extreme points of the convex set of the strong-mixing measures generating the maximal spectral type of S
    # Compute eigenvalues to impose strong positivity on the vj:
    
        # spectrum is a list of the eigenvalues of V.  If it is not given compute it
        if spectrum:
            self.W = spectrum;
        else:
            self.W = self.V.eigenvals().keys()  
        
        W = self.W
        K = [str(w).__contains__('v') for w in W].count(True) # K is the number of nontrivial conditions
        # Sanitize list of nonvariable conditions:
        for i in range(len(W)-K):
                W.pop([str(w).__contains__('v') for w in W].index(False))
        self.ExtremeK = [];               
        if K == 0:  # There no conditions, we have one point
            self.ExtremeK.append(self.V.reshape(1,self.s**2))
            sh = set((1,))
        else:
            sh = set()
            for eqn in combinations(range(len(W)),len(self.spectral_classes)-1):
                for solution in solve([W[i] for i in eqn],['v'+str(i) for i in range(1,len(self.spectral_classes))],dict = True):
                    if len(solution.keys()) == len(self.spectral_classes)-1:
                        v_temp = self.V.reshape(1,self.s**2).evalf(40, subs = solution);
                        if self.ExtremeK.count(v_temp)==0:
                            self.ExtremeK.append(v_temp);
                            sh.add(tuple(Matrix(self.v).evalf(1, subs = solution)))    
                        
                    else:
                        print('Underdetermined:', solution)
        self.ExtremeK.sort(matrix_sort)
        return sh       

    def lambda_hat(self,j,k):
        #   To find the Fourier coefficients of an extreme point of Spectral Hull:
        #     k-th coefficient of lambda_j  = self.lambda_hat(j,k)
        return round((self.ExtremeK[j]*self.SIGMA(k))[0],40)

############ Methods for Presenting and Interacting with SUBS ################

    def view(self): 
        if self.dimension == 1: # In 1-dim case output is an s x q array
            return self.sub #array([[self.instructions[j][i] for j in self.loc()] for i in range(self.s)])
        elif self.dimension == 2:  # In 2-dim case, a dictionary of arrays on shape q
            print('Sorry not supported yet.')
        else: # n-dim case
            print('Sorry not supported yet.')


######################

#   Method for drawing a square colored tile
def tile(idx,color):
    fig, (ax1,ax2) = pyplot.subplots(2,1,sharex=True)
    
    
def walk(SUB, max_iter = 0, walk_step = 1, angle_step = 60, speed = 0):
    
    
    sub = lambda k: SUB.evaluate(int(-floor(-log(k+1)/log(SUB.q))),k,0)
    k = 0;
    turtle.reset()
    turtle.pu()
    turtle.goto(-600,500)    
    turtle.pd()
    while k <= max_iter:
        print(sub(k), end="")
        if sub(k):
            turtle.forward(walk_step)
        else:
            turtle.right(angle_step)
        k+=1;

        
def alt_sum(SUB,N,shift=0,display_step=0):
    sub = lambda k: SUB.evaluate(int(-floor(-log(k+1+shift)/log(SUB.q))),k+shift,0)

    S = 0.0
    zeta = exp(2*pi*1j/SUB.s) # Primitive s-th root of unity
    
    for i in range(N):
        S = S + power(zeta,sub(i))/float(i+1)
        if display_step:
            if i % display_step == 0:
                print(S)
                
    return S
    
            
############################################################################
#  Methods for use with Substitutions

def SUBSTITUTION_PRODUCT(S,Z,iterate=1): # Computes the substitution (tensor) product of two substitutions
    if S.q != Z.q:
        return "Error - Size Mismatch"
    return SUBSTITUTION(S.q,[instr_kron(S.instructions[idx],Z.instructions[idx]) for idx in S.loc()],iterate); 
    # Need to work on product alphabet:
    
    
def ITERATE(S,Z): # Computes the iterated product (composition) of two substitutions
    if S.s != Z.s:
        return "Error - Size Mismatch"
    newq = (array(S.q)*array(Z.q)).tolist();
    pa = [i for i in location_set(newq)];
    ITER_INSTRUCTIONS = [instr_comp(S.instructions[tuple(mod(loc,Z.q))], Z.instructions[tuple(floor_divide(loc,Z.q))]) for loc in pa]
    return SUBSTITUTION(newq,ITER_INSTRUCTIONS,iterate = 1);

############################################################################
def ERG_CLASS(S):
    # Attempts to find the primitive components of the substitution.
    #   output is E_class, Trans or the ergodic classes and transient part of S
    E_class = set();
    if (matrix_power(S.substitution_matrix,S.s) != 0).all():
        E_class.add(tuple(range(S.s)));
        return E_class, set();
    
    Trans = set(tuple(range(S.s)));
    # Raise the substitutution matrix to a power to eliminate periodic behavior
    #     and make it a 0,1 matrix
    ORBIT = array(zeros(S.s));    
    for i in range(1,S.s+1):
        ORBIT = ORBIT + orbit(i,S.substitution_matrix);
    ORBIT = array(ORBIT != 0, dtype=int);
    #ORBIT = array(matrix_power(array(S.substitution_matrix,dtype='float64')/S.Q, lcm(*range(1,1+S.s))) != 0, dtype=int);
    
    # The Orbits of letter j (in range(s)) are given by the columns 
    #   ORBIT[:,j]
    
    # Orbit Sizes:
    O_count = ORBIT.sum(0)
    # As we put elements into ergodic or transient classes, update their count to be > s
    while O_count.min() <= S.s:
        # Smallest Orbits are candidates for ergodic classes: ORBIT.sum(0).argmin() is the index of first.
        #  Identify elements sharing identical orbit structure 
        if (array(ORBIT[:,O_count.argmin()]).sum()*array(ORBIT[:,O_count.argmin()]) == array([ORBIT[:,match] for match in ORBIT[:,O_count.argmin()].nonzero()[0]],dtype=bool).sum(0)).all():
            E_class.add(tuple(nonzero(ORBIT[:,O_count.argmin()])[0]));
        
        # Ergodic Class and Transient Parts identified, update O_count: just make those indices larger than s
        Trans.difference_update(set(tuple(nonzero(ORBIT[:,O_count.argmin()])[0])))
        O_count = O_count + S.s*array([(ORBIT[:,O_count.argmin()] <= ORBIT[:,j]).all() for j in range(S.s)],dtype=int);
         
    return E_class, Trans

############################################################################    
# Initiialize Substitutions

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
        
    
def draw2d(BLOCK):
    #from matplotlib import mpl,pyplot
    #import numpy as np
    #
    ## make values from -5 to 5, for this example
    #zvals = np.random.rand(100,100)*10-5
    #
    ## make a color map of fixed colors
    #cmap = mpl.colors.ListedColormap(['blue','black','red'])
    #bounds=[-6,-2,2,6]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #
    ## tell imshow about color map so that only set colors are used
    #img = pyplot.imshow(zvals,interpolation='nearest',
    #                    cmap = cmap,norm=norm)
    #
    ## make a color bar
    #pyplot.colorbar(img,cmap=cmap,
    #                norm=norm,boundaries=bounds,ticks=[-5,0,5])
    #
    #pyplot.show()
    
    ##############################################################
    
    # grid is on 10*5. If you have specified values, just change zvals
    
    zvals = rand(5,5)*6
    
    # make a color map of fixed colors
    
    cmap = mpl.colors.ListedColormap(['lightgreen','lightgray','lightpink','lightblue','lightgoldenrodyellow','maroon'])
    bounds=[-6,-4,-2,0,2,4,6]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(zvals,interpolation='nearest',
                        cmap = cmap,norm=norm)
    
    # make a color bar
    pyplot.colorbar(img,cmap=cmap,
                    norm=norm,boundaries=bounds,ticks=[0,2,4,6])
    
    pyplot.show()