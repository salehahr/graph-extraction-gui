ҩ6
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.22unknown8ד0
?
conv2d_0_block1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_0_block1/kernel
?
*conv2d_0_block1/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block1/kernel*&
_output_shapes
:*
dtype0
?
conv2d_0_block1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_0_block1/bias
y
(conv2d_0_block1/bias/Read/ReadVariableOpReadVariableOpconv2d_0_block1/bias*
_output_shapes
:*
dtype0
x
bn0_block1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebn0_block1/gamma
q
$bn0_block1/gamma/Read/ReadVariableOpReadVariableOpbn0_block1/gamma*
_output_shapes
:*
dtype0
v
bn0_block1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn0_block1/beta
o
#bn0_block1/beta/Read/ReadVariableOpReadVariableOpbn0_block1/beta*
_output_shapes
:*
dtype0
?
bn0_block1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn0_block1/moving_mean
}
*bn0_block1/moving_mean/Read/ReadVariableOpReadVariableOpbn0_block1/moving_mean*
_output_shapes
:*
dtype0
?
bn0_block1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn0_block1/moving_variance
?
.bn0_block1/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block1/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1_block1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_1_block1/kernel
?
*conv2d_1_block1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block1/kernel*&
_output_shapes
:*
dtype0
?
conv2d_1_block1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_1_block1/bias
y
(conv2d_1_block1/bias/Read/ReadVariableOpReadVariableOpconv2d_1_block1/bias*
_output_shapes
:*
dtype0
x
bn1_block1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebn1_block1/gamma
q
$bn1_block1/gamma/Read/ReadVariableOpReadVariableOpbn1_block1/gamma*
_output_shapes
:*
dtype0
v
bn1_block1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn1_block1/beta
o
#bn1_block1/beta/Read/ReadVariableOpReadVariableOpbn1_block1/beta*
_output_shapes
:*
dtype0
?
bn1_block1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn1_block1/moving_mean
}
*bn1_block1/moving_mean/Read/ReadVariableOpReadVariableOpbn1_block1/moving_mean*
_output_shapes
:*
dtype0
?
bn1_block1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn1_block1/moving_variance
?
.bn1_block1/moving_variance/Read/ReadVariableOpReadVariableOpbn1_block1/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_0_block2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_0_block2/kernel
?
*conv2d_0_block2/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block2/kernel*&
_output_shapes
:*
dtype0
?
conv2d_0_block2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_0_block2/bias
y
(conv2d_0_block2/bias/Read/ReadVariableOpReadVariableOpconv2d_0_block2/bias*
_output_shapes
:*
dtype0
x
bn0_block2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebn0_block2/gamma
q
$bn0_block2/gamma/Read/ReadVariableOpReadVariableOpbn0_block2/gamma*
_output_shapes
:*
dtype0
v
bn0_block2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn0_block2/beta
o
#bn0_block2/beta/Read/ReadVariableOpReadVariableOpbn0_block2/beta*
_output_shapes
:*
dtype0
?
bn0_block2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn0_block2/moving_mean
}
*bn0_block2/moving_mean/Read/ReadVariableOpReadVariableOpbn0_block2/moving_mean*
_output_shapes
:*
dtype0
?
bn0_block2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn0_block2/moving_variance
?
.bn0_block2/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block2/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1_block2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_1_block2/kernel
?
*conv2d_1_block2/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block2/kernel*&
_output_shapes
:*
dtype0
?
conv2d_1_block2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_1_block2/bias
y
(conv2d_1_block2/bias/Read/ReadVariableOpReadVariableOpconv2d_1_block2/bias*
_output_shapes
:*
dtype0
x
bn1_block2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebn1_block2/gamma
q
$bn1_block2/gamma/Read/ReadVariableOpReadVariableOpbn1_block2/gamma*
_output_shapes
:*
dtype0
v
bn1_block2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn1_block2/beta
o
#bn1_block2/beta/Read/ReadVariableOpReadVariableOpbn1_block2/beta*
_output_shapes
:*
dtype0
?
bn1_block2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn1_block2/moving_mean
}
*bn1_block2/moving_mean/Read/ReadVariableOpReadVariableOpbn1_block2/moving_mean*
_output_shapes
:*
dtype0
?
bn1_block2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn1_block2/moving_variance
?
.bn1_block2/moving_variance/Read/ReadVariableOpReadVariableOpbn1_block2/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_0_block3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_0_block3/kernel
?
*conv2d_0_block3/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block3/kernel*&
_output_shapes
:*
dtype0
?
conv2d_0_block3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_0_block3/bias
y
(conv2d_0_block3/bias/Read/ReadVariableOpReadVariableOpconv2d_0_block3/bias*
_output_shapes
:*
dtype0
x
bn0_block3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebn0_block3/gamma
q
$bn0_block3/gamma/Read/ReadVariableOpReadVariableOpbn0_block3/gamma*
_output_shapes
:*
dtype0
v
bn0_block3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn0_block3/beta
o
#bn0_block3/beta/Read/ReadVariableOpReadVariableOpbn0_block3/beta*
_output_shapes
:*
dtype0
?
bn0_block3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn0_block3/moving_mean
}
*bn0_block3/moving_mean/Read/ReadVariableOpReadVariableOpbn0_block3/moving_mean*
_output_shapes
:*
dtype0
?
bn0_block3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn0_block3/moving_variance
?
.bn0_block3/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block3/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1_block3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_1_block3/kernel
?
*conv2d_1_block3/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block3/kernel*&
_output_shapes
:*
dtype0
?
conv2d_1_block3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_1_block3/bias
y
(conv2d_1_block3/bias/Read/ReadVariableOpReadVariableOpconv2d_1_block3/bias*
_output_shapes
:*
dtype0
x
bn1_block3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebn1_block3/gamma
q
$bn1_block3/gamma/Read/ReadVariableOpReadVariableOpbn1_block3/gamma*
_output_shapes
:*
dtype0
v
bn1_block3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebn1_block3/beta
o
#bn1_block3/beta/Read/ReadVariableOpReadVariableOpbn1_block3/beta*
_output_shapes
:*
dtype0
?
bn1_block3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn1_block3/moving_mean
}
*bn1_block3/moving_mean/Read/ReadVariableOpReadVariableOpbn1_block3/moving_mean*
_output_shapes
:*
dtype0
?
bn1_block3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn1_block3/moving_variance
?
.bn1_block3/moving_variance/Read/ReadVariableOpReadVariableOpbn1_block3/moving_variance*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_0_block4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameconv2d_0_block4/kernel
?
*conv2d_0_block4/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block4/kernel*&
_output_shapes
:0*
dtype0
?
conv2d_0_block4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameconv2d_0_block4/bias
y
(conv2d_0_block4/bias/Read/ReadVariableOpReadVariableOpconv2d_0_block4/bias*
_output_shapes
:0*
dtype0
x
bn0_block4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_namebn0_block4/gamma
q
$bn0_block4/gamma/Read/ReadVariableOpReadVariableOpbn0_block4/gamma*
_output_shapes
:0*
dtype0
v
bn0_block4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0* 
shared_namebn0_block4/beta
o
#bn0_block4/beta/Read/ReadVariableOpReadVariableOpbn0_block4/beta*
_output_shapes
:0*
dtype0
?
bn0_block4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_namebn0_block4/moving_mean
}
*bn0_block4/moving_mean/Read/ReadVariableOpReadVariableOpbn0_block4/moving_mean*
_output_shapes
:0*
dtype0
?
bn0_block4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebn0_block4/moving_variance
?
.bn0_block4/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block4/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_1_block4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameconv2d_1_block4/kernel
?
*conv2d_1_block4/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block4/kernel*&
_output_shapes
:00*
dtype0
?
conv2d_1_block4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameconv2d_1_block4/bias
y
(conv2d_1_block4/bias/Read/ReadVariableOpReadVariableOpconv2d_1_block4/bias*
_output_shapes
:0*
dtype0
x
bn1_block4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_namebn1_block4/gamma
q
$bn1_block4/gamma/Read/ReadVariableOpReadVariableOpbn1_block4/gamma*
_output_shapes
:0*
dtype0
v
bn1_block4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0* 
shared_namebn1_block4/beta
o
#bn1_block4/beta/Read/ReadVariableOpReadVariableOpbn1_block4/beta*
_output_shapes
:0*
dtype0
?
bn1_block4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_namebn1_block4/moving_mean
}
*bn1_block4/moving_mean/Read/ReadVariableOpReadVariableOpbn1_block4/moving_mean*
_output_shapes
:0*
dtype0
?
bn1_block4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebn1_block4/moving_variance
?
.bn1_block4/moving_variance/Read/ReadVariableOpReadVariableOpbn1_block4/moving_variance*
_output_shapes
:0*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:00*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:0*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:0*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:0*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:0*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:0*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:0*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
n
accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator
g
accumulator/Read/ReadVariableOpReadVariableOpaccumulator*
_output_shapes
:*
dtype0
r
accumulator_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_1
k
!accumulator_1/Read/ReadVariableOpReadVariableOpaccumulator_1*
_output_shapes
:*
dtype0
r
accumulator_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_2
k
!accumulator_2/Read/ReadVariableOpReadVariableOpaccumulator_2*
_output_shapes
:*
dtype0
r
accumulator_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_3
k
!accumulator_3/Read/ReadVariableOpReadVariableOpaccumulator_3*
_output_shapes
:*
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
y
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_positives_2
r
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:?*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:?*
dtype0
y
true_positives_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_positives_3
r
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes	
:?*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:?*
dtype0
{
false_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_positives_2
t
%false_positives_2/Read/ReadVariableOpReadVariableOpfalse_positives_2*
_output_shapes	
:?*
dtype0
{
false_negatives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_negatives_2
t
%false_negatives_2/Read/ReadVariableOpReadVariableOpfalse_negatives_2*
_output_shapes	
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?

layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer-22
layer_with_weights-10
layer-23
layer_with_weights-11
layer-24
layer-25
layer_with_weights-12
layer-26
layer_with_weights-13
layer-27
layer-28
layer-29
layer_with_weights-14
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"layer_with_weights-16
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer_with_weights-18
%layer-36
&layer_with_weights-19
&layer-37
'layer-38
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer-42
,	optimiser

-_input
.	optimizer
/	variables
0regularization_losses
1trainable_variables
2	keras_api
3
signatures
 
 
 
R
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
?
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
R
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
R
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
h

Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
?
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
R
^	variables
_regularization_losses
`trainable_variables
a	keras_api
R
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
R
f	variables
gregularization_losses
htrainable_variables
i	keras_api
h

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
?
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
R
y	variables
zregularization_losses
{trainable_variables
|	keras_api
k

}kernel
~bias
	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api
 
 
 
?
80
91
?2
@3
A4
B5
O6
P7
V8
W9
X10
Y11
j12
k13
q14
r15
s16
t17
}18
~19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
 
?
80
91
?2
@3
O4
P5
V6
W7
j8
k9
q10
r11
}12
~13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?
 ?layer_regularization_losses
?layer_metrics
/	variables
?metrics
?non_trainable_variables
0regularization_losses
?layers
1trainable_variables
 
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
4	variables
?metrics
?non_trainable_variables
5regularization_losses
?layers
6trainable_variables
b`
VARIABLE_VALUEconv2d_0_block1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_0_block1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
?
 ?layer_regularization_losses
?layer_metrics
:	variables
?metrics
?non_trainable_variables
;regularization_losses
?layers
<trainable_variables
 
[Y
VARIABLE_VALUEbn0_block1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbn0_block1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbn0_block1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn0_block1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
A2
B3
 

?0
@1
?
 ?layer_regularization_losses
?layer_metrics
C	variables
?metrics
?non_trainable_variables
Dregularization_losses
?layers
Etrainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
G	variables
?metrics
?non_trainable_variables
Hregularization_losses
?layers
Itrainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
K	variables
?metrics
?non_trainable_variables
Lregularization_losses
?layers
Mtrainable_variables
b`
VARIABLE_VALUEconv2d_1_block1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_1_block1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
 

O0
P1
?
 ?layer_regularization_losses
?layer_metrics
Q	variables
?metrics
?non_trainable_variables
Rregularization_losses
?layers
Strainable_variables
 
[Y
VARIABLE_VALUEbn1_block1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbn1_block1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbn1_block1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn1_block1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
X2
Y3
 

V0
W1
?
 ?layer_regularization_losses
?layer_metrics
Z	variables
?metrics
?non_trainable_variables
[regularization_losses
?layers
\trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
^	variables
?metrics
?non_trainable_variables
_regularization_losses
?layers
`trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
b	variables
?metrics
?non_trainable_variables
cregularization_losses
?layers
dtrainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
f	variables
?metrics
?non_trainable_variables
gregularization_losses
?layers
htrainable_variables
b`
VARIABLE_VALUEconv2d_0_block2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_0_block2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
?
 ?layer_regularization_losses
?layer_metrics
l	variables
?metrics
?non_trainable_variables
mregularization_losses
?layers
ntrainable_variables
 
[Y
VARIABLE_VALUEbn0_block2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbn0_block2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbn0_block2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn0_block2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
s2
t3
 

q0
r1
?
 ?layer_regularization_losses
?layer_metrics
u	variables
?metrics
?non_trainable_variables
vregularization_losses
?layers
wtrainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
y	variables
?metrics
?non_trainable_variables
zregularization_losses
?layers
{trainable_variables
b`
VARIABLE_VALUEconv2d_1_block2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_1_block2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1
 

}0
~1
?
 ?layer_regularization_losses
?layer_metrics
	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
[Y
VARIABLE_VALUEbn1_block2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbn1_block2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbn1_block2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn1_block2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
b`
VARIABLE_VALUEconv2d_0_block3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_0_block3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
[Y
VARIABLE_VALUEbn0_block3/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbn0_block3/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbn0_block3/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn0_block3/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
ca
VARIABLE_VALUEconv2d_1_block3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_1_block3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
\Z
VARIABLE_VALUEbn1_block3/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbn1_block3/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn1_block3/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbn1_block3/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
ZX
VARIABLE_VALUEconv2d/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
ec
VARIABLE_VALUEbatch_normalization/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEbatch_normalization/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#batch_normalization/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
ca
VARIABLE_VALUEconv2d_0_block4/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_0_block4/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
\Z
VARIABLE_VALUEbn0_block4/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbn0_block4/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn0_block4/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbn0_block4/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
ca
VARIABLE_VALUEconv2d_1_block4/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_1_block4/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
\Z
VARIABLE_VALUEbn1_block4/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbn1_block4/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn1_block4/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbn1_block4/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
\Z
VARIABLE_VALUEconv2d_1/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_1/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_1/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_1/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_1/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_1/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
ZX
VARIABLE_VALUEconv1d/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1d/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
 
 
 
P
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?
A0
B1
X2
Y3
s4
t5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
 
 
 
 
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

X0
Y1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

s0
t1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
C
?
thresholds
?accumulator
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api
v
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
v
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
[Y
VARIABLE_VALUEaccumulator:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
 
][
VARIABLE_VALUEaccumulator_1:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
 
][
VARIABLE_VALUEaccumulator_2:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
 
][
VARIABLE_VALUEaccumulator_3:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/8/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/8/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
ca
VARIABLE_VALUEtrue_positives_3=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/9/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_2>keras_api/metrics/9/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_2>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
?
serving_default_node_pairPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
serving_default_node_posPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
serving_default_skel_imgPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_node_pairserving_default_node_posserving_default_skel_imgconv2d_0_block1/kernelconv2d_0_block1/biasbn0_block1/gammabn0_block1/betabn0_block1/moving_meanbn0_block1/moving_varianceconv2d_1_block1/kernelconv2d_1_block1/biasbn1_block1/gammabn1_block1/betabn1_block1/moving_meanbn1_block1/moving_varianceconv2d_0_block2/kernelconv2d_0_block2/biasbn0_block2/gammabn0_block2/betabn0_block2/moving_meanbn0_block2/moving_varianceconv2d_1_block2/kernelconv2d_1_block2/biasbn1_block2/gammabn1_block2/betabn1_block2/moving_meanbn1_block2/moving_varianceconv2d_0_block3/kernelconv2d_0_block3/biasbn0_block3/gammabn0_block3/betabn0_block3/moving_meanbn0_block3/moving_varianceconv2d_1_block3/kernelconv2d_1_block3/biasbn1_block3/gammabn1_block3/betabn1_block3/moving_meanbn1_block3/moving_varianceconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_0_block4/kernelconv2d_0_block4/biasbn0_block4/gammabn0_block4/betabn0_block4/moving_meanbn0_block4/moving_varianceconv2d_1_block4/kernelconv2d_1_block4/biasbn1_block4/gammabn1_block4/betabn1_block4/moving_meanbn1_block4/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv1d/kernelconv1d/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_6588
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?%
value?%B?%SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices*conv2d_0_block1/kernel/Read/ReadVariableOp(conv2d_0_block1/bias/Read/ReadVariableOp$bn0_block1/gamma/Read/ReadVariableOp#bn0_block1/beta/Read/ReadVariableOp*bn0_block1/moving_mean/Read/ReadVariableOp.bn0_block1/moving_variance/Read/ReadVariableOp*conv2d_1_block1/kernel/Read/ReadVariableOp(conv2d_1_block1/bias/Read/ReadVariableOp$bn1_block1/gamma/Read/ReadVariableOp#bn1_block1/beta/Read/ReadVariableOp*bn1_block1/moving_mean/Read/ReadVariableOp.bn1_block1/moving_variance/Read/ReadVariableOp*conv2d_0_block2/kernel/Read/ReadVariableOp(conv2d_0_block2/bias/Read/ReadVariableOp$bn0_block2/gamma/Read/ReadVariableOp#bn0_block2/beta/Read/ReadVariableOp*bn0_block2/moving_mean/Read/ReadVariableOp.bn0_block2/moving_variance/Read/ReadVariableOp*conv2d_1_block2/kernel/Read/ReadVariableOp(conv2d_1_block2/bias/Read/ReadVariableOp$bn1_block2/gamma/Read/ReadVariableOp#bn1_block2/beta/Read/ReadVariableOp*bn1_block2/moving_mean/Read/ReadVariableOp.bn1_block2/moving_variance/Read/ReadVariableOp*conv2d_0_block3/kernel/Read/ReadVariableOp(conv2d_0_block3/bias/Read/ReadVariableOp$bn0_block3/gamma/Read/ReadVariableOp#bn0_block3/beta/Read/ReadVariableOp*bn0_block3/moving_mean/Read/ReadVariableOp.bn0_block3/moving_variance/Read/ReadVariableOp*conv2d_1_block3/kernel/Read/ReadVariableOp(conv2d_1_block3/bias/Read/ReadVariableOp$bn1_block3/gamma/Read/ReadVariableOp#bn1_block3/beta/Read/ReadVariableOp*bn1_block3/moving_mean/Read/ReadVariableOp.bn1_block3/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp*conv2d_0_block4/kernel/Read/ReadVariableOp(conv2d_0_block4/bias/Read/ReadVariableOp$bn0_block4/gamma/Read/ReadVariableOp#bn0_block4/beta/Read/ReadVariableOp*bn0_block4/moving_mean/Read/ReadVariableOp.bn0_block4/moving_variance/Read/ReadVariableOp*conv2d_1_block4/kernel/Read/ReadVariableOp(conv2d_1_block4/bias/Read/ReadVariableOp$bn1_block4/gamma/Read/ReadVariableOp#bn1_block4/beta/Read/ReadVariableOp*bn1_block4/moving_mean/Read/ReadVariableOp.bn1_block4/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpaccumulator/Read/ReadVariableOp!accumulator_1/Read/ReadVariableOp!accumulator_2/Read/ReadVariableOp!accumulator_3/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp$true_positives_3/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_2/Read/ReadVariableOp%false_negatives_2/Read/ReadVariableOpConst"/device:CPU:0*a
dtypesW
U2S
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
?&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?%
value?%B?%SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOpAssignVariableOpconv2d_0_block1/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_1AssignVariableOpconv2d_0_block1/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_2AssignVariableOpbn0_block1/gamma
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_3AssignVariableOpbn0_block1/beta
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_4AssignVariableOpbn0_block1/moving_mean
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_5AssignVariableOpbn0_block1/moving_variance
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_6AssignVariableOpconv2d_1_block1/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_7AssignVariableOpconv2d_1_block1/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_8AssignVariableOpbn1_block1/gamma
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_9AssignVariableOpbn1_block1/betaIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_10AssignVariableOpbn1_block1/moving_meanIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_11AssignVariableOpbn1_block1/moving_varianceIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_12AssignVariableOpconv2d_0_block2/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_13AssignVariableOpconv2d_0_block2/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_14AssignVariableOpbn0_block2/gammaIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_15AssignVariableOpbn0_block2/betaIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_16AssignVariableOpbn0_block2/moving_meanIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_17AssignVariableOpbn0_block2/moving_varianceIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_18AssignVariableOpconv2d_1_block2/kernelIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_19AssignVariableOpconv2d_1_block2/biasIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_20AssignVariableOpbn1_block2/gammaIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_21AssignVariableOpbn1_block2/betaIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_22AssignVariableOpbn1_block2/moving_meanIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_23AssignVariableOpbn1_block2/moving_varianceIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_24AssignVariableOpconv2d_0_block3/kernelIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_25AssignVariableOpconv2d_0_block3/biasIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_26AssignVariableOpbn0_block3/gammaIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_27AssignVariableOpbn0_block3/betaIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_28AssignVariableOpbn0_block3/moving_meanIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_29AssignVariableOpbn0_block3/moving_varianceIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_30AssignVariableOpconv2d_1_block3/kernelIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_31AssignVariableOpconv2d_1_block3/biasIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_32AssignVariableOpbn1_block3/gammaIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_33AssignVariableOpbn1_block3/betaIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_34AssignVariableOpbn1_block3/moving_meanIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_35AssignVariableOpbn1_block3/moving_varianceIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_36AssignVariableOpconv2d/kernelIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_37AssignVariableOpconv2d/biasIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_38AssignVariableOpbatch_normalization/gammaIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_39AssignVariableOpbatch_normalization/betaIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
q
AssignVariableOp_40AssignVariableOpbatch_normalization/moving_meanIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_41AssignVariableOp#batch_normalization/moving_varianceIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_42AssignVariableOpconv2d_0_block4/kernelIdentity_43"/device:CPU:0*
dtype0
W
Identity_44IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_43AssignVariableOpconv2d_0_block4/biasIdentity_44"/device:CPU:0*
dtype0
W
Identity_45IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_44AssignVariableOpbn0_block4/gammaIdentity_45"/device:CPU:0*
dtype0
W
Identity_46IdentityRestoreV2:45"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_45AssignVariableOpbn0_block4/betaIdentity_46"/device:CPU:0*
dtype0
W
Identity_47IdentityRestoreV2:46"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_46AssignVariableOpbn0_block4/moving_meanIdentity_47"/device:CPU:0*
dtype0
W
Identity_48IdentityRestoreV2:47"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_47AssignVariableOpbn0_block4/moving_varianceIdentity_48"/device:CPU:0*
dtype0
W
Identity_49IdentityRestoreV2:48"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_48AssignVariableOpconv2d_1_block4/kernelIdentity_49"/device:CPU:0*
dtype0
W
Identity_50IdentityRestoreV2:49"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_49AssignVariableOpconv2d_1_block4/biasIdentity_50"/device:CPU:0*
dtype0
W
Identity_51IdentityRestoreV2:50"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_50AssignVariableOpbn1_block4/gammaIdentity_51"/device:CPU:0*
dtype0
W
Identity_52IdentityRestoreV2:51"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_51AssignVariableOpbn1_block4/betaIdentity_52"/device:CPU:0*
dtype0
W
Identity_53IdentityRestoreV2:52"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_52AssignVariableOpbn1_block4/moving_meanIdentity_53"/device:CPU:0*
dtype0
W
Identity_54IdentityRestoreV2:53"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_53AssignVariableOpbn1_block4/moving_varianceIdentity_54"/device:CPU:0*
dtype0
W
Identity_55IdentityRestoreV2:54"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_54AssignVariableOpconv2d_1/kernelIdentity_55"/device:CPU:0*
dtype0
W
Identity_56IdentityRestoreV2:55"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_55AssignVariableOpconv2d_1/biasIdentity_56"/device:CPU:0*
dtype0
W
Identity_57IdentityRestoreV2:56"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_56AssignVariableOpbatch_normalization_1/gammaIdentity_57"/device:CPU:0*
dtype0
W
Identity_58IdentityRestoreV2:57"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_57AssignVariableOpbatch_normalization_1/betaIdentity_58"/device:CPU:0*
dtype0
W
Identity_59IdentityRestoreV2:58"/device:CPU:0*
T0*
_output_shapes
:
s
AssignVariableOp_58AssignVariableOp!batch_normalization_1/moving_meanIdentity_59"/device:CPU:0*
dtype0
W
Identity_60IdentityRestoreV2:59"/device:CPU:0*
T0*
_output_shapes
:
w
AssignVariableOp_59AssignVariableOp%batch_normalization_1/moving_varianceIdentity_60"/device:CPU:0*
dtype0
W
Identity_61IdentityRestoreV2:60"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_60AssignVariableOpconv1d/kernelIdentity_61"/device:CPU:0*
dtype0
W
Identity_62IdentityRestoreV2:61"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_61AssignVariableOpconv1d/biasIdentity_62"/device:CPU:0*
dtype0
W
Identity_63IdentityRestoreV2:62"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_62AssignVariableOptotalIdentity_63"/device:CPU:0*
dtype0
W
Identity_64IdentityRestoreV2:63"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_63AssignVariableOpcountIdentity_64"/device:CPU:0*
dtype0
W
Identity_65IdentityRestoreV2:64"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_64AssignVariableOpaccumulatorIdentity_65"/device:CPU:0*
dtype0
W
Identity_66IdentityRestoreV2:65"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_65AssignVariableOpaccumulator_1Identity_66"/device:CPU:0*
dtype0
W
Identity_67IdentityRestoreV2:66"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_66AssignVariableOpaccumulator_2Identity_67"/device:CPU:0*
dtype0
W
Identity_68IdentityRestoreV2:67"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_67AssignVariableOpaccumulator_3Identity_68"/device:CPU:0*
dtype0
W
Identity_69IdentityRestoreV2:68"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_68AssignVariableOptotal_1Identity_69"/device:CPU:0*
dtype0
W
Identity_70IdentityRestoreV2:69"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_69AssignVariableOpcount_1Identity_70"/device:CPU:0*
dtype0
W
Identity_71IdentityRestoreV2:70"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_70AssignVariableOptrue_positivesIdentity_71"/device:CPU:0*
dtype0
W
Identity_72IdentityRestoreV2:71"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_71AssignVariableOpfalse_positivesIdentity_72"/device:CPU:0*
dtype0
W
Identity_73IdentityRestoreV2:72"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_72AssignVariableOptrue_positives_1Identity_73"/device:CPU:0*
dtype0
W
Identity_74IdentityRestoreV2:73"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_73AssignVariableOpfalse_negativesIdentity_74"/device:CPU:0*
dtype0
W
Identity_75IdentityRestoreV2:74"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_74AssignVariableOptrue_positives_2Identity_75"/device:CPU:0*
dtype0
W
Identity_76IdentityRestoreV2:75"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_75AssignVariableOptrue_negativesIdentity_76"/device:CPU:0*
dtype0
W
Identity_77IdentityRestoreV2:76"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_76AssignVariableOpfalse_positives_1Identity_77"/device:CPU:0*
dtype0
W
Identity_78IdentityRestoreV2:77"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_77AssignVariableOpfalse_negatives_1Identity_78"/device:CPU:0*
dtype0
W
Identity_79IdentityRestoreV2:78"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_78AssignVariableOptrue_positives_3Identity_79"/device:CPU:0*
dtype0
W
Identity_80IdentityRestoreV2:79"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_79AssignVariableOptrue_negatives_1Identity_80"/device:CPU:0*
dtype0
W
Identity_81IdentityRestoreV2:80"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_80AssignVariableOpfalse_positives_2Identity_81"/device:CPU:0*
dtype0
W
Identity_82IdentityRestoreV2:81"/device:CPU:0*
T0*
_output_shapes
:
c
AssignVariableOp_81AssignVariableOpfalse_negatives_2Identity_82"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_83Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: Ғ+
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_8280

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
b
(__inference_summation_layer_call_fn_7158
inputs_0
inputs_1
inputs_2
identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:???????????:???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7820

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
.__inference_conv2d_1_block3_layer_call_fn_8116

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?2
%__inference_EdgeNN_layer_call_fn_3034
skel_img
node_pos
	node_pairH
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block2/FusedBatchNormV3?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block2/FusedBatchNormV3?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn0_block3/FusedBatchNormV3?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn1_block3/FusedBatchNormV3?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn0_block4/FusedBatchNormV3?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn1_block4/FusedBatchNormV3?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
skel_img:[W
1
_output_shapes
:???????????
"
_user_specified_name
node_pos:\X
1
_output_shapes
:???????????
#
_user_specified_name	node_pair
?

?
'__inference_conv2d_1_layer_call_fn_8832

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
b
F__inference_relu0_block2_layer_call_and_return_conditional_losses_7723

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8850

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
)__inference_bn1_block2_layer_call_fn_7892

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
I__inference_conv2d_0_block1_layer_call_and_return_conditional_losses_7168

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_1_layer_call_fn_8940

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?

?
.__inference_conv2d_0_block3_layer_call_fn_7942

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
.__inference_conv2d_0_block4_layer_call_fn_8484

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
.__inference_conv2d_1_block4_layer_call_fn_8658

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8152

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn0_block2_layer_call_fn_7664

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8188

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_2_layer_call_fn_8459

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7610

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn0_block1_layer_call_fn_7322

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?7
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6154
skel_img
node_pos
	node_pairH
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?bn0_block1/AssignNewValue?bn0_block1/AssignNewValue_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?bn0_block2/AssignNewValue?bn0_block2/AssignNewValue_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?bn0_block3/AssignNewValue?bn0_block3/AssignNewValue_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?bn0_block4/AssignNewValue?bn0_block4/AssignNewValue_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?bn1_block1/AssignNewValue?bn1_block1/AssignNewValue_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?bn1_block2/AssignNewValue?bn1_block2/AssignNewValue_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?bn1_block3/AssignNewValue?bn1_block3/AssignNewValue_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?bn1_block4/AssignNewValue?bn1_block4/AssignNewValue_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block1/FusedBatchNormV3?
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue?
bn0_block1/AssignNewValue_1AssignVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource,bn0_block1/FusedBatchNormV3:batch_variance:0-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block1/FusedBatchNormV3?
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue?
bn1_block1/AssignNewValue_1AssignVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource,bn1_block1/FusedBatchNormV3:batch_variance:0-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block2/FusedBatchNormV3?
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue?
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block2/FusedBatchNormV3?
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue?
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block3/FusedBatchNormV3?
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue?
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block3/FusedBatchNormV3?
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue?
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block4/FusedBatchNormV3?
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue?
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block4/FusedBatchNormV3?
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue?
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_126
bn0_block1/AssignNewValuebn0_block1/AssignNewValue2:
bn0_block1/AssignNewValue_1bn0_block1/AssignNewValue_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_126
bn0_block2/AssignNewValuebn0_block2/AssignNewValue2:
bn0_block2/AssignNewValue_1bn0_block2/AssignNewValue_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_126
bn0_block3/AssignNewValuebn0_block3/AssignNewValue2:
bn0_block3/AssignNewValue_1bn0_block3/AssignNewValue_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_126
bn0_block4/AssignNewValuebn0_block4/AssignNewValue2:
bn0_block4/AssignNewValue_1bn0_block4/AssignNewValue_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_126
bn1_block1/AssignNewValuebn1_block1/AssignNewValue2:
bn1_block1/AssignNewValue_1bn1_block1/AssignNewValue_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_126
bn1_block2/AssignNewValuebn1_block2/AssignNewValue2:
bn1_block2/AssignNewValue_1bn1_block2/AssignNewValue_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_126
bn1_block3/AssignNewValuebn1_block3/AssignNewValue2:
bn1_block3/AssignNewValue_1bn1_block3/AssignNewValue_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_126
bn1_block4/AssignNewValuebn1_block4/AssignNewValue2:
bn1_block4/AssignNewValue_1bn1_block4/AssignNewValue_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
skel_img:[W
1
_output_shapes
:???????????
"
_user_specified_name
node_pos:\X
1
_output_shapes
:???????????
#
_user_specified_name	node_pair
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8454

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
G
+__inference_relu0_block4_layer_call_fn_8638

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
d
H__inference_relu_C3_block4_layer_call_and_return_conditional_losses_8981

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7438

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
O
3__inference_global_max_pooling2d_layer_call_fn_9024

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices?
MaxMaxinputsMax/reduction_indices:output:0*
T0*8
_output_shapes&
$:"??????????????????*
	keep_dims(2
Maxq
IdentityIdentityMax:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?9
__inference__wrapped_model_1194
skel_img
node_pos
	node_pairO
5edgenn_conv2d_0_block1_conv2d_readvariableop_resource:D
6edgenn_conv2d_0_block1_biasadd_readvariableop_resource:7
)edgenn_bn0_block1_readvariableop_resource:9
+edgenn_bn0_block1_readvariableop_1_resource:H
:edgenn_bn0_block1_fusedbatchnormv3_readvariableop_resource:J
<edgenn_bn0_block1_fusedbatchnormv3_readvariableop_1_resource:O
5edgenn_conv2d_1_block1_conv2d_readvariableop_resource:D
6edgenn_conv2d_1_block1_biasadd_readvariableop_resource:7
)edgenn_bn1_block1_readvariableop_resource:9
+edgenn_bn1_block1_readvariableop_1_resource:H
:edgenn_bn1_block1_fusedbatchnormv3_readvariableop_resource:J
<edgenn_bn1_block1_fusedbatchnormv3_readvariableop_1_resource:O
5edgenn_conv2d_0_block2_conv2d_readvariableop_resource:D
6edgenn_conv2d_0_block2_biasadd_readvariableop_resource:7
)edgenn_bn0_block2_readvariableop_resource:9
+edgenn_bn0_block2_readvariableop_1_resource:H
:edgenn_bn0_block2_fusedbatchnormv3_readvariableop_resource:J
<edgenn_bn0_block2_fusedbatchnormv3_readvariableop_1_resource:O
5edgenn_conv2d_1_block2_conv2d_readvariableop_resource:D
6edgenn_conv2d_1_block2_biasadd_readvariableop_resource:7
)edgenn_bn1_block2_readvariableop_resource:9
+edgenn_bn1_block2_readvariableop_1_resource:H
:edgenn_bn1_block2_fusedbatchnormv3_readvariableop_resource:J
<edgenn_bn1_block2_fusedbatchnormv3_readvariableop_1_resource:O
5edgenn_conv2d_0_block3_conv2d_readvariableop_resource:D
6edgenn_conv2d_0_block3_biasadd_readvariableop_resource:7
)edgenn_bn0_block3_readvariableop_resource:9
+edgenn_bn0_block3_readvariableop_1_resource:H
:edgenn_bn0_block3_fusedbatchnormv3_readvariableop_resource:J
<edgenn_bn0_block3_fusedbatchnormv3_readvariableop_1_resource:O
5edgenn_conv2d_1_block3_conv2d_readvariableop_resource:D
6edgenn_conv2d_1_block3_biasadd_readvariableop_resource:7
)edgenn_bn1_block3_readvariableop_resource:9
+edgenn_bn1_block3_readvariableop_1_resource:H
:edgenn_bn1_block3_fusedbatchnormv3_readvariableop_resource:J
<edgenn_bn1_block3_fusedbatchnormv3_readvariableop_1_resource:F
,edgenn_conv2d_conv2d_readvariableop_resource:;
-edgenn_conv2d_biasadd_readvariableop_resource:@
2edgenn_batch_normalization_readvariableop_resource:B
4edgenn_batch_normalization_readvariableop_1_resource:Q
Cedgenn_batch_normalization_fusedbatchnormv3_readvariableop_resource:S
Eedgenn_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:O
5edgenn_conv2d_0_block4_conv2d_readvariableop_resource:0D
6edgenn_conv2d_0_block4_biasadd_readvariableop_resource:07
)edgenn_bn0_block4_readvariableop_resource:09
+edgenn_bn0_block4_readvariableop_1_resource:0H
:edgenn_bn0_block4_fusedbatchnormv3_readvariableop_resource:0J
<edgenn_bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0O
5edgenn_conv2d_1_block4_conv2d_readvariableop_resource:00D
6edgenn_conv2d_1_block4_biasadd_readvariableop_resource:07
)edgenn_bn1_block4_readvariableop_resource:09
+edgenn_bn1_block4_readvariableop_1_resource:0H
:edgenn_bn1_block4_fusedbatchnormv3_readvariableop_resource:0J
<edgenn_bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.edgenn_conv2d_1_conv2d_readvariableop_resource:00=
/edgenn_conv2d_1_biasadd_readvariableop_resource:0B
4edgenn_batch_normalization_1_readvariableop_resource:0D
6edgenn_batch_normalization_1_readvariableop_1_resource:0S
Eedgenn_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0U
Gedgenn_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0O
9edgenn_conv1d_conv1d_expanddims_1_readvariableop_resource:0N
@edgenn_conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp?<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?)EdgeNN/batch_normalization/ReadVariableOp?+EdgeNN/batch_normalization/ReadVariableOp_1?<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?+EdgeNN/batch_normalization_1/ReadVariableOp?-EdgeNN/batch_normalization_1/ReadVariableOp_1?1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn0_block1/ReadVariableOp?"EdgeNN/bn0_block1/ReadVariableOp_1?1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn0_block2/ReadVariableOp?"EdgeNN/bn0_block2/ReadVariableOp_1?1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn0_block3/ReadVariableOp?"EdgeNN/bn0_block3/ReadVariableOp_1?1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn0_block4/ReadVariableOp?"EdgeNN/bn0_block4/ReadVariableOp_1?1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn1_block1/ReadVariableOp?"EdgeNN/bn1_block1/ReadVariableOp_1?1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn1_block2/ReadVariableOp?"EdgeNN/bn1_block2/ReadVariableOp_1?1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn1_block3/ReadVariableOp?"EdgeNN/bn1_block3/ReadVariableOp_1?1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp?3EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1? EdgeNN/bn1_block4/ReadVariableOp?"EdgeNN/bn1_block4/ReadVariableOp_1?0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp?7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?$EdgeNN/conv2d/BiasAdd/ReadVariableOp?#EdgeNN/conv2d/Conv2D/ReadVariableOp?-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp?-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp?-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp?-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp?&EdgeNN/conv2d_1/BiasAdd/ReadVariableOp?%EdgeNN/conv2d_1/Conv2D/ReadVariableOp?-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp?-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp?-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp?-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp?,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp?
EdgeNN/summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
EdgeNN/summation/concat/axis?
EdgeNN/summation/concatConcatV2skel_imgnode_pos	node_pair%EdgeNN/summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
EdgeNN/summation/concat?
&EdgeNN/summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&EdgeNN/summation/Sum/reduction_indices?
EdgeNN/summation/SumSum EdgeNN/summation/concat:output:0/EdgeNN/summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
EdgeNN/summation/Sum?
,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp?
EdgeNN/conv2d_0_block1/Conv2DConv2DEdgeNN/summation/Sum:output:04EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block1/Conv2D?
-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_0_block1/BiasAddBiasAdd&EdgeNN/conv2d_0_block1/Conv2D:output:05EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
EdgeNN/conv2d_0_block1/BiasAdd?
 EdgeNN/bn0_block1/ReadVariableOpReadVariableOp)edgenn_bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn0_block1/ReadVariableOp?
"EdgeNN/bn0_block1/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn0_block1/ReadVariableOp_1?
1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn0_block1/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block1/BiasAdd:output:0(EdgeNN/bn0_block1/ReadVariableOp:value:0*EdgeNN/bn0_block1/ReadVariableOp_1:value:09EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn0_block1/FusedBatchNormV3?
EdgeNN/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
EdgeNN/concatenate/concat/axis?
EdgeNN/concatenate/concatConcatV2&EdgeNN/bn0_block1/FusedBatchNormV3:y:0	node_pair'EdgeNN/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
EdgeNN/concatenate/concat?
EdgeNN/relu0_block1/ReluRelu"EdgeNN/concatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
EdgeNN/relu0_block1/Relu?
,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp?
EdgeNN/conv2d_1_block1/Conv2DConv2D&EdgeNN/relu0_block1/Relu:activations:04EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block1/Conv2D?
-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_1_block1/BiasAddBiasAdd&EdgeNN/conv2d_1_block1/Conv2D:output:05EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
EdgeNN/conv2d_1_block1/BiasAdd?
 EdgeNN/bn1_block1/ReadVariableOpReadVariableOp)edgenn_bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn1_block1/ReadVariableOp?
"EdgeNN/bn1_block1/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn1_block1/ReadVariableOp_1?
1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn1_block1/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block1/BiasAdd:output:0(EdgeNN/bn1_block1/ReadVariableOp:value:0*EdgeNN/bn1_block1/ReadVariableOp_1:value:09EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn1_block1/FusedBatchNormV3?
 EdgeNN/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2"
 EdgeNN/concatenate_1/concat/axis?
EdgeNN/concatenate_1/concatConcatV2&EdgeNN/bn1_block1/FusedBatchNormV3:y:0	node_pair)EdgeNN/concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
EdgeNN/concatenate_1/concat?
EdgeNN/relu1_block1/ReluRelu$EdgeNN/concatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
EdgeNN/relu1_block1/Relu?
EdgeNN/max_pooling2d/MaxPoolMaxPool&EdgeNN/relu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
EdgeNN/max_pooling2d/MaxPool?
,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp?
EdgeNN/conv2d_0_block2/Conv2DConv2D%EdgeNN/max_pooling2d/MaxPool:output:04EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block2/Conv2D?
-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_0_block2/BiasAddBiasAdd&EdgeNN/conv2d_0_block2/Conv2D:output:05EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
EdgeNN/conv2d_0_block2/BiasAdd?
 EdgeNN/bn0_block2/ReadVariableOpReadVariableOp)edgenn_bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn0_block2/ReadVariableOp?
"EdgeNN/bn0_block2/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn0_block2/ReadVariableOp_1?
1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn0_block2/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block2/BiasAdd:output:0(EdgeNN/bn0_block2/ReadVariableOp:value:0*EdgeNN/bn0_block2/ReadVariableOp_1:value:09EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn0_block2/FusedBatchNormV3?
EdgeNN/relu0_block2/ReluRelu&EdgeNN/bn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
EdgeNN/relu0_block2/Relu?
,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp?
EdgeNN/conv2d_1_block2/Conv2DConv2D&EdgeNN/relu0_block2/Relu:activations:04EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block2/Conv2D?
-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_1_block2/BiasAddBiasAdd&EdgeNN/conv2d_1_block2/Conv2D:output:05EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
EdgeNN/conv2d_1_block2/BiasAdd?
 EdgeNN/bn1_block2/ReadVariableOpReadVariableOp)edgenn_bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn1_block2/ReadVariableOp?
"EdgeNN/bn1_block2/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn1_block2/ReadVariableOp_1?
1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn1_block2/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block2/BiasAdd:output:0(EdgeNN/bn1_block2/ReadVariableOp:value:0*EdgeNN/bn1_block2/ReadVariableOp_1:value:09EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn1_block2/FusedBatchNormV3?
EdgeNN/relu1_block2/ReluRelu&EdgeNN/bn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
EdgeNN/relu1_block2/Relu?
EdgeNN/max_pooling2d_1/MaxPoolMaxPool&EdgeNN/relu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2 
EdgeNN/max_pooling2d_1/MaxPool?
,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp?
EdgeNN/conv2d_0_block3/Conv2DConv2D'EdgeNN/max_pooling2d_1/MaxPool:output:04EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block3/Conv2D?
-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_0_block3/BiasAddBiasAdd&EdgeNN/conv2d_0_block3/Conv2D:output:05EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2 
EdgeNN/conv2d_0_block3/BiasAdd?
 EdgeNN/bn0_block3/ReadVariableOpReadVariableOp)edgenn_bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn0_block3/ReadVariableOp?
"EdgeNN/bn0_block3/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn0_block3/ReadVariableOp_1?
1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn0_block3/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block3/BiasAdd:output:0(EdgeNN/bn0_block3/ReadVariableOp:value:0*EdgeNN/bn0_block3/ReadVariableOp_1:value:09EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn0_block3/FusedBatchNormV3?
EdgeNN/relu0_block3/ReluRelu&EdgeNN/bn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
EdgeNN/relu0_block3/Relu?
,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp?
EdgeNN/conv2d_1_block3/Conv2DConv2D&EdgeNN/relu0_block3/Relu:activations:04EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block3/Conv2D?
-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_1_block3/BiasAddBiasAdd&EdgeNN/conv2d_1_block3/Conv2D:output:05EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2 
EdgeNN/conv2d_1_block3/BiasAdd?
 EdgeNN/bn1_block3/ReadVariableOpReadVariableOp)edgenn_bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn1_block3/ReadVariableOp?
"EdgeNN/bn1_block3/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn1_block3/ReadVariableOp_1?
1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn1_block3/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block3/BiasAdd:output:0(EdgeNN/bn1_block3/ReadVariableOp:value:0*EdgeNN/bn1_block3/ReadVariableOp_1:value:09EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn1_block3/FusedBatchNormV3?
EdgeNN/relu1_block3/ReluRelu&EdgeNN/bn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
EdgeNN/relu1_block3/Relu?
#EdgeNN/conv2d/Conv2D/ReadVariableOpReadVariableOp,edgenn_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#EdgeNN/conv2d/Conv2D/ReadVariableOp?
EdgeNN/conv2d/Conv2DConv2D&EdgeNN/relu1_block3/Relu:activations:0+EdgeNN/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
EdgeNN/conv2d/Conv2D?
$EdgeNN/conv2d/BiasAdd/ReadVariableOpReadVariableOp-edgenn_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$EdgeNN/conv2d/BiasAdd/ReadVariableOp?
EdgeNN/conv2d/BiasAddBiasAddEdgeNN/conv2d/Conv2D:output:0,EdgeNN/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
EdgeNN/conv2d/BiasAdd?
)EdgeNN/batch_normalization/ReadVariableOpReadVariableOp2edgenn_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02+
)EdgeNN/batch_normalization/ReadVariableOp?
+EdgeNN/batch_normalization/ReadVariableOp_1ReadVariableOp4edgenn_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02-
+EdgeNN/batch_normalization/ReadVariableOp_1?
:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpCedgenn_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02<
:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp?
<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEedgenn_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02>
<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
+EdgeNN/batch_normalization/FusedBatchNormV3FusedBatchNormV3EdgeNN/conv2d/BiasAdd:output:01EdgeNN/batch_normalization/ReadVariableOp:value:03EdgeNN/batch_normalization/ReadVariableOp_1:value:0BEdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0DEdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2-
+EdgeNN/batch_normalization/FusedBatchNormV3?
EdgeNN/relu_C3_block3/ReluRelu/EdgeNN/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
EdgeNN/relu_C3_block3/Relu?
EdgeNN/max_pooling2d_2/MaxPoolMaxPool(EdgeNN/relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2 
EdgeNN/max_pooling2d_2/MaxPool?
,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp?
EdgeNN/conv2d_0_block4/Conv2DConv2D'EdgeNN/max_pooling2d_2/MaxPool:output:04EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block4/Conv2D?
-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_0_block4/BiasAddBiasAdd&EdgeNN/conv2d_0_block4/Conv2D:output:05EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02 
EdgeNN/conv2d_0_block4/BiasAdd?
 EdgeNN/bn0_block4/ReadVariableOpReadVariableOp)edgenn_bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02"
 EdgeNN/bn0_block4/ReadVariableOp?
"EdgeNN/bn0_block4/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02$
"EdgeNN/bn0_block4/ReadVariableOp_1?
1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype023
1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn0_block4/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block4/BiasAdd:output:0(EdgeNN/bn0_block4/ReadVariableOp:value:0*EdgeNN/bn0_block4/ReadVariableOp_1:value:09EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn0_block4/FusedBatchNormV3?
EdgeNN/relu0_block4/ReluRelu&EdgeNN/bn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
EdgeNN/relu0_block4/Relu?
,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02.
,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp?
EdgeNN/conv2d_1_block4/Conv2DConv2D&EdgeNN/relu0_block4/Relu:activations:04EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block4/Conv2D?
-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_1_block4/BiasAddBiasAdd&EdgeNN/conv2d_1_block4/Conv2D:output:05EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02 
EdgeNN/conv2d_1_block4/BiasAdd?
 EdgeNN/bn1_block4/ReadVariableOpReadVariableOp)edgenn_bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02"
 EdgeNN/bn1_block4/ReadVariableOp?
"EdgeNN/bn1_block4/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02$
"EdgeNN/bn1_block4/ReadVariableOp_1?
1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype023
1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp?
3EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
"EdgeNN/bn1_block4/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block4/BiasAdd:output:0(EdgeNN/bn1_block4/ReadVariableOp:value:0*EdgeNN/bn1_block4/ReadVariableOp_1:value:09EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2$
"EdgeNN/bn1_block4/FusedBatchNormV3?
EdgeNN/relu1_block4/ReluRelu&EdgeNN/bn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
EdgeNN/relu1_block4/Relu?
%EdgeNN/conv2d_1/Conv2D/ReadVariableOpReadVariableOp.edgenn_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%EdgeNN/conv2d_1/Conv2D/ReadVariableOp?
EdgeNN/conv2d_1/Conv2DConv2D&EdgeNN/relu1_block4/Relu:activations:0-EdgeNN/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
EdgeNN/conv2d_1/Conv2D?
&EdgeNN/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp/edgenn_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&EdgeNN/conv2d_1/BiasAdd/ReadVariableOp?
EdgeNN/conv2d_1/BiasAddBiasAddEdgeNN/conv2d_1/Conv2D:output:0.EdgeNN/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
EdgeNN/conv2d_1/BiasAdd?
+EdgeNN/batch_normalization_1/ReadVariableOpReadVariableOp4edgenn_batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02-
+EdgeNN/batch_normalization_1/ReadVariableOp?
-EdgeNN/batch_normalization_1/ReadVariableOp_1ReadVariableOp6edgenn_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02/
-EdgeNN/batch_normalization_1/ReadVariableOp_1?
<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpEedgenn_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02>
<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGedgenn_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02@
>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
-EdgeNN/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 EdgeNN/conv2d_1/BiasAdd:output:03EdgeNN/batch_normalization_1/ReadVariableOp:value:05EdgeNN/batch_normalization_1/ReadVariableOp_1:value:0DEdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0FEdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2/
-EdgeNN/batch_normalization_1/FusedBatchNormV3?
EdgeNN/relu_C3_block4/ReluRelu1EdgeNN/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
EdgeNN/relu_C3_block4/Relu?
EdgeNN/max_pooling2d_3/MaxPoolMaxPool(EdgeNN/relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2 
EdgeNN/max_pooling2d_3/MaxPool?
1EdgeNN/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1EdgeNN/global_max_pooling2d/Max/reduction_indices?
EdgeNN/global_max_pooling2d/MaxMax'EdgeNN/max_pooling2d_3/MaxPool:output:0:EdgeNN/global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2!
EdgeNN/global_max_pooling2d/Max?
#EdgeNN/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#EdgeNN/conv1d/conv1d/ExpandDims/dim?
EdgeNN/conv1d/conv1d/ExpandDims
ExpandDims(EdgeNN/global_max_pooling2d/Max:output:0,EdgeNN/conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02!
EdgeNN/conv1d/conv1d/ExpandDims?
0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9edgenn_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype022
0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
%EdgeNN/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%EdgeNN/conv1d/conv1d/ExpandDims_1/dim?
!EdgeNN/conv1d/conv1d/ExpandDims_1
ExpandDims8EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0.EdgeNN/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02#
!EdgeNN/conv1d/conv1d/ExpandDims_1?
EdgeNN/conv1d/conv1d/ShapeShape(EdgeNN/conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
EdgeNN/conv1d/conv1d/Shape?
(EdgeNN/conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(EdgeNN/conv1d/conv1d/strided_slice/stack?
*EdgeNN/conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*EdgeNN/conv1d/conv1d/strided_slice/stack_1?
*EdgeNN/conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*EdgeNN/conv1d/conv1d/strided_slice/stack_2?
"EdgeNN/conv1d/conv1d/strided_sliceStridedSlice#EdgeNN/conv1d/conv1d/Shape:output:01EdgeNN/conv1d/conv1d/strided_slice/stack:output:03EdgeNN/conv1d/conv1d/strided_slice/stack_1:output:03EdgeNN/conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"EdgeNN/conv1d/conv1d/strided_slice?
"EdgeNN/conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2$
"EdgeNN/conv1d/conv1d/Reshape/shape?
EdgeNN/conv1d/conv1d/ReshapeReshape(EdgeNN/conv1d/conv1d/ExpandDims:output:0+EdgeNN/conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
EdgeNN/conv1d/conv1d/Reshape?
EdgeNN/conv1d/conv1d/Conv2DConv2D%EdgeNN/conv1d/conv1d/Reshape:output:0*EdgeNN/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
EdgeNN/conv1d/conv1d/Conv2D?
$EdgeNN/conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2&
$EdgeNN/conv1d/conv1d/concat/values_1?
 EdgeNN/conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 EdgeNN/conv1d/conv1d/concat/axis?
EdgeNN/conv1d/conv1d/concatConcatV2+EdgeNN/conv1d/conv1d/strided_slice:output:0-EdgeNN/conv1d/conv1d/concat/values_1:output:0)EdgeNN/conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
EdgeNN/conv1d/conv1d/concat?
EdgeNN/conv1d/conv1d/Reshape_1Reshape$EdgeNN/conv1d/conv1d/Conv2D:output:0$EdgeNN/conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2 
EdgeNN/conv1d/conv1d/Reshape_1?
EdgeNN/conv1d/conv1d/SqueezeSqueeze'EdgeNN/conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
EdgeNN/conv1d/conv1d/Squeeze?
&EdgeNN/conv1d/squeeze_batch_dims/ShapeShape%EdgeNN/conv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2(
&EdgeNN/conv1d/squeeze_batch_dims/Shape?
4EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack?
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????28
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_1?
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_2?
.EdgeNN/conv1d/squeeze_batch_dims/strided_sliceStridedSlice/EdgeNN/conv1d/squeeze_batch_dims/Shape:output:0=EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack:output:0?EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_1:output:0?EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask20
.EdgeNN/conv1d/squeeze_batch_dims/strided_slice?
.EdgeNN/conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      20
.EdgeNN/conv1d/squeeze_batch_dims/Reshape/shape?
(EdgeNN/conv1d/squeeze_batch_dims/ReshapeReshape%EdgeNN/conv1d/conv1d/Squeeze:output:07EdgeNN/conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2*
(EdgeNN/conv1d/squeeze_batch_dims/Reshape?
7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp@edgenn_conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
(EdgeNN/conv1d/squeeze_batch_dims/BiasAddBiasAdd1EdgeNN/conv1d/squeeze_batch_dims/Reshape:output:0?EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2*
(EdgeNN/conv1d/squeeze_batch_dims/BiasAdd?
0EdgeNN/conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      22
0EdgeNN/conv1d/squeeze_batch_dims/concat/values_1?
,EdgeNN/conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,EdgeNN/conv1d/squeeze_batch_dims/concat/axis?
'EdgeNN/conv1d/squeeze_batch_dims/concatConcatV27EdgeNN/conv1d/squeeze_batch_dims/strided_slice:output:09EdgeNN/conv1d/squeeze_batch_dims/concat/values_1:output:05EdgeNN/conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'EdgeNN/conv1d/squeeze_batch_dims/concat?
*EdgeNN/conv1d/squeeze_batch_dims/Reshape_1Reshape1EdgeNN/conv1d/squeeze_batch_dims/BiasAdd:output:00EdgeNN/conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2,
*EdgeNN/conv1d/squeeze_batch_dims/Reshape_1?
EdgeNN/conv1d/SigmoidSigmoid3EdgeNN/conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
EdgeNN/conv1d/Sigmoid?
&EdgeNN/tf.compat.v1.squeeze/adj_outputSqueezeEdgeNN/conv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2(
&EdgeNN/tf.compat.v1.squeeze/adj_output?
IdentityIdentity/EdgeNN/tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp;^EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp=^EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*^EdgeNN/batch_normalization/ReadVariableOp,^EdgeNN/batch_normalization/ReadVariableOp_1=^EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?^EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1,^EdgeNN/batch_normalization_1/ReadVariableOp.^EdgeNN/batch_normalization_1/ReadVariableOp_12^EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block1/ReadVariableOp#^EdgeNN/bn0_block1/ReadVariableOp_12^EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block2/ReadVariableOp#^EdgeNN/bn0_block2/ReadVariableOp_12^EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block3/ReadVariableOp#^EdgeNN/bn0_block3/ReadVariableOp_12^EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block4/ReadVariableOp#^EdgeNN/bn0_block4/ReadVariableOp_12^EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block1/ReadVariableOp#^EdgeNN/bn1_block1/ReadVariableOp_12^EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block2/ReadVariableOp#^EdgeNN/bn1_block2/ReadVariableOp_12^EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block3/ReadVariableOp#^EdgeNN/bn1_block3/ReadVariableOp_12^EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block4/ReadVariableOp#^EdgeNN/bn1_block4/ReadVariableOp_11^EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp8^EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp%^EdgeNN/conv2d/BiasAdd/ReadVariableOp$^EdgeNN/conv2d/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp'^EdgeNN/conv2d_1/BiasAdd/ReadVariableOp&^EdgeNN/conv2d_1/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp2|
<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_12V
)EdgeNN/batch_normalization/ReadVariableOp)EdgeNN/batch_normalization/ReadVariableOp2Z
+EdgeNN/batch_normalization/ReadVariableOp_1+EdgeNN/batch_normalization/ReadVariableOp_12|
<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12Z
+EdgeNN/batch_normalization_1/ReadVariableOp+EdgeNN/batch_normalization_1/ReadVariableOp2^
-EdgeNN/batch_normalization_1/ReadVariableOp_1-EdgeNN/batch_normalization_1/ReadVariableOp_12f
1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn0_block1/ReadVariableOp EdgeNN/bn0_block1/ReadVariableOp2H
"EdgeNN/bn0_block1/ReadVariableOp_1"EdgeNN/bn0_block1/ReadVariableOp_12f
1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn0_block2/ReadVariableOp EdgeNN/bn0_block2/ReadVariableOp2H
"EdgeNN/bn0_block2/ReadVariableOp_1"EdgeNN/bn0_block2/ReadVariableOp_12f
1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn0_block3/ReadVariableOp EdgeNN/bn0_block3/ReadVariableOp2H
"EdgeNN/bn0_block3/ReadVariableOp_1"EdgeNN/bn0_block3/ReadVariableOp_12f
1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn0_block4/ReadVariableOp EdgeNN/bn0_block4/ReadVariableOp2H
"EdgeNN/bn0_block4/ReadVariableOp_1"EdgeNN/bn0_block4/ReadVariableOp_12f
1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn1_block1/ReadVariableOp EdgeNN/bn1_block1/ReadVariableOp2H
"EdgeNN/bn1_block1/ReadVariableOp_1"EdgeNN/bn1_block1/ReadVariableOp_12f
1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn1_block2/ReadVariableOp EdgeNN/bn1_block2/ReadVariableOp2H
"EdgeNN/bn1_block2/ReadVariableOp_1"EdgeNN/bn1_block2/ReadVariableOp_12f
1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn1_block3/ReadVariableOp EdgeNN/bn1_block3/ReadVariableOp2H
"EdgeNN/bn1_block3/ReadVariableOp_1"EdgeNN/bn1_block3/ReadVariableOp_12f
1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp2j
3EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_13EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_12D
 EdgeNN/bn1_block4/ReadVariableOp EdgeNN/bn1_block4/ReadVariableOp2H
"EdgeNN/bn1_block4/ReadVariableOp_1"EdgeNN/bn1_block4/ReadVariableOp_12d
0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp2r
7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2L
$EdgeNN/conv2d/BiasAdd/ReadVariableOp$EdgeNN/conv2d/BiasAdd/ReadVariableOp2J
#EdgeNN/conv2d/Conv2D/ReadVariableOp#EdgeNN/conv2d/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp2P
&EdgeNN/conv2d_1/BiasAdd/ReadVariableOp&EdgeNN/conv2d_1/BiasAdd/ReadVariableOp2N
%EdgeNN/conv2d_1/Conv2D/ReadVariableOp%EdgeNN/conv2d_1/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp2^
-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp2\
,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
skel_img:[W
1
_output_shapes
:???????????
"
_user_specified_name
node_pos:\X
1
_output_shapes
:???????????
#
_user_specified_name	node_pair
?
G
+__inference_relu0_block3_layer_call_fn_8096

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_bn1_block4_layer_call_fn_8784

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7978

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7196

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7907

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
s
G__inference_concatenate_1_layer_call_and_return_conditional_losses_7517
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
)__inference_bn0_block2_layer_call_fn_7700

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_3_layer_call_fn_9006

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8134

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8520

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
I__inference_conv2d_1_block3_layer_call_and_return_conditional_losses_8106

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_bn0_block2_layer_call_fn_7718

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_relu1_block3_layer_call_fn_8270

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_1_layer_call_fn_8922

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
ߜ
?7
%__inference_EdgeNN_layer_call_fn_9636
inputs_0
inputs_1
inputs_2H
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?bn0_block1/AssignNewValue?bn0_block1/AssignNewValue_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?bn0_block2/AssignNewValue?bn0_block2/AssignNewValue_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?bn0_block3/AssignNewValue?bn0_block3/AssignNewValue_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?bn0_block4/AssignNewValue?bn0_block4/AssignNewValue_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?bn1_block1/AssignNewValue?bn1_block1/AssignNewValue_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?bn1_block2/AssignNewValue?bn1_block2/AssignNewValue_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?bn1_block3/AssignNewValue?bn1_block3/AssignNewValue_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?bn1_block4/AssignNewValue?bn1_block4/AssignNewValue_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block1/FusedBatchNormV3?
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue?
bn0_block1/AssignNewValue_1AssignVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource,bn0_block1/FusedBatchNormV3:batch_variance:0-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block1/FusedBatchNormV3?
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue?
bn1_block1/AssignNewValue_1AssignVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource,bn1_block1/FusedBatchNormV3:batch_variance:0-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block2/FusedBatchNormV3?
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue?
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block2/FusedBatchNormV3?
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue?
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block3/FusedBatchNormV3?
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue?
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block3/FusedBatchNormV3?
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue?
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block4/FusedBatchNormV3?
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue?
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block4/FusedBatchNormV3?
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue?
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_126
bn0_block1/AssignNewValuebn0_block1/AssignNewValue2:
bn0_block1/AssignNewValue_1bn0_block1/AssignNewValue_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_126
bn0_block2/AssignNewValuebn0_block2/AssignNewValue2:
bn0_block2/AssignNewValue_1bn0_block2/AssignNewValue_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_126
bn0_block3/AssignNewValuebn0_block3/AssignNewValue2:
bn0_block3/AssignNewValue_1bn0_block3/AssignNewValue_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_126
bn0_block4/AssignNewValuebn0_block4/AssignNewValue2:
bn0_block4/AssignNewValue_1bn0_block4/AssignNewValue_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_126
bn1_block1/AssignNewValuebn1_block1/AssignNewValue2:
bn1_block1/AssignNewValue_1bn1_block1/AssignNewValue_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_126
bn1_block2/AssignNewValuebn1_block2/AssignNewValue2:
bn1_block2/AssignNewValue_1bn1_block2/AssignNewValue_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_126
bn1_block3/AssignNewValuebn1_block3/AssignNewValue2:
bn1_block3/AssignNewValue_1bn1_block3/AssignNewValue_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_126
bn1_block4/AssignNewValuebn1_block4/AssignNewValue2:
bn1_block4/AssignNewValue_1bn1_block4/AssignNewValue_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7592

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?1
?
@__inference_conv1d_layer_call_and_return_conditional_losses_9068

inputsA
+conv1d_expanddims_1_readvariableop_resource:0@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"conv1d/ExpandDims_1/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/ExpandDims_1f
conv1d/ShapeShapeconv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/Shape?
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack?
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
conv1d/strided_slice/stack_1?
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2?
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice?
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/Reshape/shape?
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/Reshape?
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/Conv2D?
conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/concat/values_1s
conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/concat/axis?
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat?
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/Reshape_1?
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape?
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack?
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(squeeze_batch_dims/strided_slice/stack_1?
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice?
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2"
 squeeze_batch_dims/Reshape/shape?
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/Reshape?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/BiasAdd?
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1?
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
squeeze_batch_dims/concat/axis?
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2
squeeze_batch_dims/Reshape_1~
SigmoidSigmoid%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
)__inference_bn0_block3_layer_call_fn_8068

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_bn0_block1_layer_call_fn_7286

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
V
*__inference_concatenate_layer_call_fn_7336
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7996

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
I__inference_conv2d_1_block1_layer_call_and_return_conditional_losses_7356

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7628

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8556

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
j
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9018

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices?
MaxMaxinputsMax/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
Maxh
IdentityIdentityMax:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
G
+__inference_relu1_block4_layer_call_fn_8812

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7766

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_relu1_block3_layer_call_and_return_conditional_losses_8265

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7646

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_bn1_block4_layer_call_fn_8766

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
b
F__inference_relu0_block3_layer_call_and_return_conditional_losses_8091

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7402

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7784

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn0_block3_layer_call_fn_8050

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_1_layer_call_fn_7922

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_6588
	node_pair
node_pos
skel_img!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:$

unknown_41:0

unknown_42:0

unknown_43:0

unknown_44:0

unknown_45:0

unknown_46:0$

unknown_47:00

unknown_48:0

unknown_49:0

unknown_50:0

unknown_51:0

unknown_52:0$

unknown_53:00

unknown_54:0

unknown_55:0

unknown_56:0

unknown_57:0

unknown_58:0 

unknown_59:0

unknown_60:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallskel_imgnode_pos	node_pairunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_11942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:???????????
#
_user_specified_name	node_pair:[W
1
_output_shapes
:???????????
"
_user_specified_name
node_pos:[W
1
_output_shapes
:???????????
"
_user_specified_name
skel_img
?
?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8730

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?

?
.__inference_conv2d_0_block2_layer_call_fn_7574

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_bn1_block3_layer_call_fn_8206

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn0_block3_layer_call_fn_8032

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8362

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_bn0_block4_layer_call_fn_8610

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
)__inference_bn1_block3_layer_call_fn_8224

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_batch_normalization_layer_call_fn_8380

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
.__inference_conv2d_1_block2_layer_call_fn_7748

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
q
E__inference_concatenate_layer_call_and_return_conditional_losses_7329
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
)__inference_bn1_block1_layer_call_fn_7510

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_relu0_block1_layer_call_fn_7346

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_relu1_block2_layer_call_and_return_conditional_losses_7897

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
I__inference_conv2d_1_block4_layer_call_and_return_conditional_losses_8648

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?1
?
%__inference_conv1d_layer_call_fn_9106

inputsA
+conv1d_expanddims_1_readvariableop_resource:0@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"conv1d/ExpandDims_1/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/ExpandDims_1f
conv1d/ShapeShapeconv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/Shape?
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack?
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
conv1d/strided_slice/stack_1?
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2?
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice?
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/Reshape/shape?
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/Reshape?
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/Conv2D?
conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/concat/values_1s
conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/concat/axis?
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat?
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/Reshape_1?
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape?
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack?
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(squeeze_batch_dims/strided_slice/stack_1?
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice?
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2"
 squeeze_batch_dims/Reshape/shape?
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/Reshape?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/BiasAdd?
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1?
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
squeeze_batch_dims/concat/axis?
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2
squeeze_batch_dims/Reshape_1~
SigmoidSigmoid%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?	
}
C__inference_summation_layer_call_and_return_conditional_losses_7128
inputs_0
inputs_1
inputs_2
identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:???????????:???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8694

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
H
,__inference_max_pooling2d_layer_call_fn_7549

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_relu1_block1_layer_call_and_return_conditional_losses_7529

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
I__inference_conv2d_1_block2_layer_call_and_return_conditional_losses_7738

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_bn1_block3_layer_call_fn_8242

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
b
(__inference_summation_layer_call_fn_7148
inputs_0
inputs_1
inputs_2
identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:???????????:???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
?
)__inference_bn0_block1_layer_call_fn_7268

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8344

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_bn0_block4_layer_call_fn_8592

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
2__inference_batch_normalization_layer_call_fn_8416

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_bn1_block1_layer_call_fn_7492

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8170

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
I__inference_conv2d_0_block4_layer_call_and_return_conditional_losses_8474

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7420

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
%__inference_conv2d_layer_call_fn_8290

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8712

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
I
-__inference_relu_C3_block4_layer_call_fn_8986

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
)__inference_bn0_block3_layer_call_fn_8086

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
X
,__inference_concatenate_1_layer_call_fn_7524
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?

?
.__inference_conv2d_0_block1_layer_call_fn_7178

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_relu1_block1_layer_call_fn_7534

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_relu1_block2_layer_call_fn_7902

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_8014

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8449

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_batch_normalization_layer_call_fn_8434

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?2
%__inference_EdgeNN_layer_call_fn_9371
inputs_0
inputs_1
inputs_2H
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block2/FusedBatchNormV3?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block2/FusedBatchNormV3?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn0_block3/FusedBatchNormV3?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn1_block3/FusedBatchNormV3?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn0_block4/FusedBatchNormV3?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn1_block4/FusedBatchNormV3?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8904

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7214

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn1_block3_layer_call_fn_8260

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8502

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8538

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7960

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_relu0_block4_layer_call_and_return_conditional_losses_8633

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
)__inference_bn0_block4_layer_call_fn_8574

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
)__inference_bn1_block4_layer_call_fn_8802

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8326

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn1_block1_layer_call_fn_7474

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?2
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6853
inputs_0
inputs_1
inputs_2H
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block2/FusedBatchNormV3?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block2/FusedBatchNormV3?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn0_block3/FusedBatchNormV3?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn1_block3/FusedBatchNormV3?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn0_block4/FusedBatchNormV3?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn1_block4/FusedBatchNormV3?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7250

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
}
C__inference_summation_layer_call_and_return_conditional_losses_7138
inputs_0
inputs_1
inputs_2
identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:???????????:???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
O
3__inference_global_max_pooling2d_layer_call_fn_9030

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices?
MaxMaxinputsMax/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
Maxh
IdentityIdentityMax:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8868

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
G
+__inference_relu0_block2_layer_call_fn_7728

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_bn0_block2_layer_call_fn_7682

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8886

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
??
?7
%__inference_EdgeNN_layer_call_fn_5624
skel_img
node_pos
	node_pairH
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?bn0_block1/AssignNewValue?bn0_block1/AssignNewValue_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?bn0_block2/AssignNewValue?bn0_block2/AssignNewValue_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?bn0_block3/AssignNewValue?bn0_block3/AssignNewValue_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?bn0_block4/AssignNewValue?bn0_block4/AssignNewValue_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?bn1_block1/AssignNewValue?bn1_block1/AssignNewValue_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?bn1_block2/AssignNewValue?bn1_block2/AssignNewValue_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?bn1_block3/AssignNewValue?bn1_block3/AssignNewValue_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?bn1_block4/AssignNewValue?bn1_block4/AssignNewValue_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block1/FusedBatchNormV3?
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue?
bn0_block1/AssignNewValue_1AssignVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource,bn0_block1/FusedBatchNormV3:batch_variance:0-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block1/FusedBatchNormV3?
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue?
bn1_block1/AssignNewValue_1AssignVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource,bn1_block1/FusedBatchNormV3:batch_variance:0-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block2/FusedBatchNormV3?
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue?
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block2/FusedBatchNormV3?
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue?
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block3/FusedBatchNormV3?
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue?
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block3/FusedBatchNormV3?
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue?
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block4/FusedBatchNormV3?
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue?
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block4/FusedBatchNormV3?
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue?
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_126
bn0_block1/AssignNewValuebn0_block1/AssignNewValue2:
bn0_block1/AssignNewValue_1bn0_block1/AssignNewValue_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_126
bn0_block2/AssignNewValuebn0_block2/AssignNewValue2:
bn0_block2/AssignNewValue_1bn0_block2/AssignNewValue_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_126
bn0_block3/AssignNewValuebn0_block3/AssignNewValue2:
bn0_block3/AssignNewValue_1bn0_block3/AssignNewValue_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_126
bn0_block4/AssignNewValuebn0_block4/AssignNewValue2:
bn0_block4/AssignNewValue_1bn0_block4/AssignNewValue_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_126
bn1_block1/AssignNewValuebn1_block1/AssignNewValue2:
bn1_block1/AssignNewValue_1bn1_block1/AssignNewValue_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_126
bn1_block2/AssignNewValuebn1_block2/AssignNewValue2:
bn1_block2/AssignNewValue_1bn1_block2/AssignNewValue_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_126
bn1_block3/AssignNewValuebn1_block3/AssignNewValue2:
bn1_block3/AssignNewValue_1bn1_block3/AssignNewValue_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_126
bn1_block4/AssignNewValuebn1_block4/AssignNewValue2:
bn1_block4/AssignNewValue_1bn1_block4/AssignNewValue_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
skel_img:[W
1
_output_shapes
:???????????
"
_user_specified_name
node_pos:\X
1
_output_shapes
:???????????
#
_user_specified_name	node_pair
?
?
2__inference_batch_normalization_layer_call_fn_8398

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn0_block4_layer_call_fn_8628

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_1_layer_call_fn_7917

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn0_block1_layer_call_fn_7304

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
I
-__inference_relu_C3_block3_layer_call_fn_8444

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_1_layer_call_fn_8976

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
??
?2
@__inference_EdgeNN_layer_call_and_return_conditional_losses_5889
skel_img
node_pos
	node_pairH
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn0_block2/FusedBatchNormV3?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
bn1_block2/FusedBatchNormV3?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn0_block3/FusedBatchNormV3?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
bn1_block3/FusedBatchNormV3?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn0_block4/FusedBatchNormV3?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
bn1_block4/FusedBatchNormV3?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
skel_img:[W
1
_output_shapes
:???????????
"
_user_specified_name
node_pos:\X
1
_output_shapes
:???????????
#
_user_specified_name	node_pair
?
H
,__inference_max_pooling2d_layer_call_fn_7554

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8996

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7802

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8822

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
j
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9012

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices?
MaxMaxinputsMax/reduction_indices:output:0*
T0*8
_output_shapes&
$:"??????????????????*
	keep_dims(2
Maxq
IdentityIdentityMax:output:0*
T0*8
_output_shapes&
$:"??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_conv2d_0_block2_layer_call_and_return_conditional_losses_7564

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7544

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_bn1_block2_layer_call_fn_7838

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8308

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_relu_C3_block3_layer_call_and_return_conditional_losses_8439

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_3_layer_call_fn_9001

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7912

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_relu0_block1_layer_call_and_return_conditional_losses_7341

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_relu1_block4_layer_call_and_return_conditional_losses_8807

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  0:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
)__inference_bn1_block2_layer_call_fn_7874

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_1_layer_call_fn_8958

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  0
 
_user_specified_nameinputs
?
?
)__inference_bn1_block2_layer_call_fn_7856

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7232

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_2_layer_call_fn_8464

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
.__inference_conv2d_1_block1_layer_call_fn_7366

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8676

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
??
?7
@__inference_EdgeNN_layer_call_and_return_conditional_losses_7118
inputs_0
inputs_1
inputs_2H
.conv2d_0_block1_conv2d_readvariableop_resource:=
/conv2d_0_block1_biasadd_readvariableop_resource:0
"bn0_block1_readvariableop_resource:2
$bn0_block1_readvariableop_1_resource:A
3bn0_block1_fusedbatchnormv3_readvariableop_resource:C
5bn0_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block1_conv2d_readvariableop_resource:=
/conv2d_1_block1_biasadd_readvariableop_resource:0
"bn1_block1_readvariableop_resource:2
$bn1_block1_readvariableop_1_resource:A
3bn1_block1_fusedbatchnormv3_readvariableop_resource:C
5bn1_block1_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block2_conv2d_readvariableop_resource:=
/conv2d_0_block2_biasadd_readvariableop_resource:0
"bn0_block2_readvariableop_resource:2
$bn0_block2_readvariableop_1_resource:A
3bn0_block2_fusedbatchnormv3_readvariableop_resource:C
5bn0_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block2_conv2d_readvariableop_resource:=
/conv2d_1_block2_biasadd_readvariableop_resource:0
"bn1_block2_readvariableop_resource:2
$bn1_block2_readvariableop_1_resource:A
3bn1_block2_fusedbatchnormv3_readvariableop_resource:C
5bn1_block2_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block3_conv2d_readvariableop_resource:=
/conv2d_0_block3_biasadd_readvariableop_resource:0
"bn0_block3_readvariableop_resource:2
$bn0_block3_readvariableop_1_resource:A
3bn0_block3_fusedbatchnormv3_readvariableop_resource:C
5bn0_block3_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_1_block3_conv2d_readvariableop_resource:=
/conv2d_1_block3_biasadd_readvariableop_resource:0
"bn1_block3_readvariableop_resource:2
$bn1_block3_readvariableop_1_resource:A
3bn1_block3_fusedbatchnormv3_readvariableop_resource:C
5bn1_block3_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
.conv2d_0_block4_conv2d_readvariableop_resource:0=
/conv2d_0_block4_biasadd_readvariableop_resource:00
"bn0_block4_readvariableop_resource:02
$bn0_block4_readvariableop_1_resource:0A
3bn0_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn0_block4_fusedbatchnormv3_readvariableop_1_resource:0H
.conv2d_1_block4_conv2d_readvariableop_resource:00=
/conv2d_1_block4_biasadd_readvariableop_resource:00
"bn1_block4_readvariableop_resource:02
$bn1_block4_readvariableop_1_resource:0A
3bn1_block4_fusedbatchnormv3_readvariableop_resource:0C
5bn1_block4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_1_conv2d_readvariableop_resource:006
(conv2d_1_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0H
2conv1d_conv1d_expanddims_1_readvariableop_resource:0G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?bn0_block1/AssignNewValue?bn0_block1/AssignNewValue_1?*bn0_block1/FusedBatchNormV3/ReadVariableOp?,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?bn0_block1/ReadVariableOp?bn0_block1/ReadVariableOp_1?bn0_block2/AssignNewValue?bn0_block2/AssignNewValue_1?*bn0_block2/FusedBatchNormV3/ReadVariableOp?,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?bn0_block2/ReadVariableOp?bn0_block2/ReadVariableOp_1?bn0_block3/AssignNewValue?bn0_block3/AssignNewValue_1?*bn0_block3/FusedBatchNormV3/ReadVariableOp?,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?bn0_block3/ReadVariableOp?bn0_block3/ReadVariableOp_1?bn0_block4/AssignNewValue?bn0_block4/AssignNewValue_1?*bn0_block4/FusedBatchNormV3/ReadVariableOp?,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?bn0_block4/ReadVariableOp?bn0_block4/ReadVariableOp_1?bn1_block1/AssignNewValue?bn1_block1/AssignNewValue_1?*bn1_block1/FusedBatchNormV3/ReadVariableOp?,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?bn1_block1/ReadVariableOp?bn1_block1/ReadVariableOp_1?bn1_block2/AssignNewValue?bn1_block2/AssignNewValue_1?*bn1_block2/FusedBatchNormV3/ReadVariableOp?,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?bn1_block2/ReadVariableOp?bn1_block2/ReadVariableOp_1?bn1_block3/AssignNewValue?bn1_block3/AssignNewValue_1?*bn1_block3/FusedBatchNormV3/ReadVariableOp?,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?bn1_block3/ReadVariableOp?bn1_block3/ReadVariableOp_1?bn1_block4/AssignNewValue?bn1_block4/AssignNewValue_1?*bn1_block4/FusedBatchNormV3/ReadVariableOp?,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?bn1_block4/ReadVariableOp?bn1_block4/ReadVariableOp_1?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?&conv2d_0_block1/BiasAdd/ReadVariableOp?%conv2d_0_block1/Conv2D/ReadVariableOp?&conv2d_0_block2/BiasAdd/ReadVariableOp?%conv2d_0_block2/Conv2D/ReadVariableOp?&conv2d_0_block3/BiasAdd/ReadVariableOp?%conv2d_0_block3/Conv2D/ReadVariableOp?&conv2d_0_block4/BiasAdd/ReadVariableOp?%conv2d_0_block4/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?&conv2d_1_block1/BiasAdd/ReadVariableOp?%conv2d_1_block1/Conv2D/ReadVariableOp?&conv2d_1_block2/BiasAdd/ReadVariableOp?%conv2d_1_block2/Conv2D/ReadVariableOp?&conv2d_1_block3/BiasAdd/ReadVariableOp?%conv2d_1_block3/Conv2D/ReadVariableOp?&conv2d_1_block4/BiasAdd/ReadVariableOp?%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
summation/concat/axis?
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
summation/concat?
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
summation/Sum/reduction_indices?
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:???????????*
	keep_dims(2
summation/Sum?
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOp?
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block1/Conv2D?
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOp?
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block1/BiasAdd?
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp?
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1?
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOp?
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1?
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block1/FusedBatchNormV3?
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue?
bn0_block1/AssignNewValue_1AssignVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource,bn0_block1/FusedBatchNormV3:batch_variance:0-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate/concat?
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:???????????2
relu0_block1/Relu?
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOp?
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block1/Conv2D?
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOp?
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block1/BiasAdd?
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp?
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1?
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOp?
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1?
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block1/FusedBatchNormV3?
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue?
bn1_block1/AssignNewValue_1AssignVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource,bn1_block1/FusedBatchNormV3:batch_variance:0-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:???????????2
relu1_block1/Relu?
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOp?
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_block2/Conv2D?
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOp?
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_block2/BiasAdd?
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp?
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1?
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOp?
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1?
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block2/FusedBatchNormV3?
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue?
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1?
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_block2/Relu?
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOp?
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_block2/Conv2D?
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOp?
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_block2/BiasAdd?
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp?
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1?
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOp?
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1?
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block2/FusedBatchNormV3?
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue?
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1?
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_block2/Relu?
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOp?
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2D?
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOp?
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_0_block3/BiasAdd?
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp?
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1?
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOp?
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1?
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block3/FusedBatchNormV3?
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue?
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1?
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu0_block3/Relu?
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOp?
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2D?
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOp?
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_1_block3/BiasAdd?
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp?
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1?
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOp?
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1?
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block3/FusedBatchNormV3?
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue?
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1?
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu1_block3/Relu?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@2
relu_C3_block3/Relu?
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOp?
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2D?
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOp?
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_0_block4/BiasAdd?
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp?
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1?
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOp?
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1?
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn0_block4/FusedBatchNormV3?
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue?
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1?
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu0_block4/Relu?
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOp?
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2D?
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOp?
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1_block4/BiasAdd?
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp?
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1?
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOp?
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1?
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1_block4/FusedBatchNormV3?
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue?
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1?
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu1_block4/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  0*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  02
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  02
relu_C3_block4/Relu?
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:?????????0*
	keep_dims(2
global_max_pooling2d/Max?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:?????????02
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape?
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack?
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#conv1d/conv1d/strided_slice/stack_1?
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2?
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice?
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      0   2
conv1d/conv1d/Reshape/shape?
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????02
conv1d/conv1d/Reshape?
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D?
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1?
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/concat/axis?
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat?
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/conv1d/Reshape_1?
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape?
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stack?
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????21
/conv1d/squeeze_batch_dims/strided_slice/stack_1?
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2?
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice?
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2)
'conv1d/squeeze_batch_dims/Reshape/shape?
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/Reshape?
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp?
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2#
!conv1d/squeeze_batch_dims/BiasAdd?
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1?
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%conv1d/squeeze_batch_dims/concat/axis?
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concat?
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2%
#conv1d/squeeze_batch_dims/Reshape_1?
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d/Sigmoid?
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output?
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_126
bn0_block1/AssignNewValuebn0_block1/AssignNewValue2:
bn0_block1/AssignNewValue_1bn0_block1/AssignNewValue_12X
*bn0_block1/FusedBatchNormV3/ReadVariableOp*bn0_block1/FusedBatchNormV3/ReadVariableOp2\
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1,bn0_block1/FusedBatchNormV3/ReadVariableOp_126
bn0_block1/ReadVariableOpbn0_block1/ReadVariableOp2:
bn0_block1/ReadVariableOp_1bn0_block1/ReadVariableOp_126
bn0_block2/AssignNewValuebn0_block2/AssignNewValue2:
bn0_block2/AssignNewValue_1bn0_block2/AssignNewValue_12X
*bn0_block2/FusedBatchNormV3/ReadVariableOp*bn0_block2/FusedBatchNormV3/ReadVariableOp2\
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1,bn0_block2/FusedBatchNormV3/ReadVariableOp_126
bn0_block2/ReadVariableOpbn0_block2/ReadVariableOp2:
bn0_block2/ReadVariableOp_1bn0_block2/ReadVariableOp_126
bn0_block3/AssignNewValuebn0_block3/AssignNewValue2:
bn0_block3/AssignNewValue_1bn0_block3/AssignNewValue_12X
*bn0_block3/FusedBatchNormV3/ReadVariableOp*bn0_block3/FusedBatchNormV3/ReadVariableOp2\
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1,bn0_block3/FusedBatchNormV3/ReadVariableOp_126
bn0_block3/ReadVariableOpbn0_block3/ReadVariableOp2:
bn0_block3/ReadVariableOp_1bn0_block3/ReadVariableOp_126
bn0_block4/AssignNewValuebn0_block4/AssignNewValue2:
bn0_block4/AssignNewValue_1bn0_block4/AssignNewValue_12X
*bn0_block4/FusedBatchNormV3/ReadVariableOp*bn0_block4/FusedBatchNormV3/ReadVariableOp2\
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1,bn0_block4/FusedBatchNormV3/ReadVariableOp_126
bn0_block4/ReadVariableOpbn0_block4/ReadVariableOp2:
bn0_block4/ReadVariableOp_1bn0_block4/ReadVariableOp_126
bn1_block1/AssignNewValuebn1_block1/AssignNewValue2:
bn1_block1/AssignNewValue_1bn1_block1/AssignNewValue_12X
*bn1_block1/FusedBatchNormV3/ReadVariableOp*bn1_block1/FusedBatchNormV3/ReadVariableOp2\
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1,bn1_block1/FusedBatchNormV3/ReadVariableOp_126
bn1_block1/ReadVariableOpbn1_block1/ReadVariableOp2:
bn1_block1/ReadVariableOp_1bn1_block1/ReadVariableOp_126
bn1_block2/AssignNewValuebn1_block2/AssignNewValue2:
bn1_block2/AssignNewValue_1bn1_block2/AssignNewValue_12X
*bn1_block2/FusedBatchNormV3/ReadVariableOp*bn1_block2/FusedBatchNormV3/ReadVariableOp2\
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1,bn1_block2/FusedBatchNormV3/ReadVariableOp_126
bn1_block2/ReadVariableOpbn1_block2/ReadVariableOp2:
bn1_block2/ReadVariableOp_1bn1_block2/ReadVariableOp_126
bn1_block3/AssignNewValuebn1_block3/AssignNewValue2:
bn1_block3/AssignNewValue_1bn1_block3/AssignNewValue_12X
*bn1_block3/FusedBatchNormV3/ReadVariableOp*bn1_block3/FusedBatchNormV3/ReadVariableOp2\
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1,bn1_block3/FusedBatchNormV3/ReadVariableOp_126
bn1_block3/ReadVariableOpbn1_block3/ReadVariableOp2:
bn1_block3/ReadVariableOp_1bn1_block3/ReadVariableOp_126
bn1_block4/AssignNewValuebn1_block4/AssignNewValue2:
bn1_block4/AssignNewValue_1bn1_block4/AssignNewValue_12X
*bn1_block4/FusedBatchNormV3/ReadVariableOp*bn1_block4/FusedBatchNormV3/ReadVariableOp2\
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1,bn1_block4/FusedBatchNormV3/ReadVariableOp_126
bn1_block4/ReadVariableOpbn1_block4/ReadVariableOp2:
bn1_block4/ReadVariableOp_1bn1_block4/ReadVariableOp_12V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2P
&conv2d_0_block1/BiasAdd/ReadVariableOp&conv2d_0_block1/BiasAdd/ReadVariableOp2N
%conv2d_0_block1/Conv2D/ReadVariableOp%conv2d_0_block1/Conv2D/ReadVariableOp2P
&conv2d_0_block2/BiasAdd/ReadVariableOp&conv2d_0_block2/BiasAdd/ReadVariableOp2N
%conv2d_0_block2/Conv2D/ReadVariableOp%conv2d_0_block2/Conv2D/ReadVariableOp2P
&conv2d_0_block3/BiasAdd/ReadVariableOp&conv2d_0_block3/BiasAdd/ReadVariableOp2N
%conv2d_0_block3/Conv2D/ReadVariableOp%conv2d_0_block3/Conv2D/ReadVariableOp2P
&conv2d_0_block4/BiasAdd/ReadVariableOp&conv2d_0_block4/BiasAdd/ReadVariableOp2N
%conv2d_0_block4/Conv2D/ReadVariableOp%conv2d_0_block4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&conv2d_1_block1/BiasAdd/ReadVariableOp&conv2d_1_block1/BiasAdd/ReadVariableOp2N
%conv2d_1_block1/Conv2D/ReadVariableOp%conv2d_1_block1/Conv2D/ReadVariableOp2P
&conv2d_1_block2/BiasAdd/ReadVariableOp&conv2d_1_block2/BiasAdd/ReadVariableOp2N
%conv2d_1_block2/Conv2D/ReadVariableOp%conv2d_1_block2/Conv2D/ReadVariableOp2P
&conv2d_1_block3/BiasAdd/ReadVariableOp&conv2d_1_block3/BiasAdd/ReadVariableOp2N
%conv2d_1_block3/Conv2D/ReadVariableOp%conv2d_1_block3/Conv2D/ReadVariableOp2P
&conv2d_1_block4/BiasAdd/ReadVariableOp&conv2d_1_block4/BiasAdd/ReadVariableOp2N
%conv2d_1_block4/Conv2D/ReadVariableOp%conv2d_1_block4/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/2
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7539

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_bn1_block4_layer_call_fn_8748

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
I__inference_conv2d_0_block3_layer_call_and_return_conditional_losses_7932

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_bn1_block1_layer_call_fn_7456

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8991

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7384

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs"?-
saver_filename:0
Identity:0Identity_838"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
	node_pair<
serving_default_node_pair:0???????????
G
node_pos;
serving_default_node_pos:0???????????
G
skel_img;
serving_default_skel_img:0???????????H
tf.compat.v1.squeeze0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?

layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer-22
layer_with_weights-10
layer-23
layer_with_weights-11
layer-24
layer-25
layer_with_weights-12
layer-26
layer_with_weights-13
layer-27
layer-28
layer-29
layer_with_weights-14
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"layer_with_weights-16
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer_with_weights-18
%layer-36
&layer_with_weights-19
&layer-37
'layer-38
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer-42
,	optimiser

-_input
.	optimizer
/	variables
0regularization_losses
1trainable_variables
2	keras_api
3
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

}kernel
~bias
	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
"
	optimizer
 "
trackable_list_wrapper
"
	optimizer
?
80
91
?2
@3
A4
B5
O6
P7
V8
W9
X10
Y11
j12
k13
q14
r15
s16
t17
}18
~19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
80
91
?2
@3
O4
P5
V6
W7
j8
k9
q10
r11
}12
~13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
/	variables
?metrics
?non_trainable_variables
0regularization_losses
?layers
1trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
4	variables
?metrics
?non_trainable_variables
5regularization_losses
?layers
6trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_0_block1/kernel
": 2conv2d_0_block1/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
:	variables
?metrics
?non_trainable_variables
;regularization_losses
?layers
<trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn0_block1/gamma
:2bn0_block1/beta
&:$ (2bn0_block1/moving_mean
*:( (2bn0_block1/moving_variance
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
C	variables
?metrics
?non_trainable_variables
Dregularization_losses
?layers
Etrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
G	variables
?metrics
?non_trainable_variables
Hregularization_losses
?layers
Itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
K	variables
?metrics
?non_trainable_variables
Lregularization_losses
?layers
Mtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_1_block1/kernel
": 2conv2d_1_block1/bias
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
Q	variables
?metrics
?non_trainable_variables
Rregularization_losses
?layers
Strainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn1_block1/gamma
:2bn1_block1/beta
&:$ (2bn1_block1/moving_mean
*:( (2bn1_block1/moving_variance
<
V0
W1
X2
Y3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
Z	variables
?metrics
?non_trainable_variables
[regularization_losses
?layers
\trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
^	variables
?metrics
?non_trainable_variables
_regularization_losses
?layers
`trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
b	variables
?metrics
?non_trainable_variables
cregularization_losses
?layers
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
f	variables
?metrics
?non_trainable_variables
gregularization_losses
?layers
htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_0_block2/kernel
": 2conv2d_0_block2/bias
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
l	variables
?metrics
?non_trainable_variables
mregularization_losses
?layers
ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn0_block2/gamma
:2bn0_block2/beta
&:$ (2bn0_block2/moving_mean
*:( (2bn0_block2/moving_variance
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
u	variables
?metrics
?non_trainable_variables
vregularization_losses
?layers
wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
y	variables
?metrics
?non_trainable_variables
zregularization_losses
?layers
{trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_1_block2/kernel
": 2conv2d_1_block2/bias
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn1_block2/gamma
:2bn1_block2/beta
&:$ (2bn1_block2/moving_mean
*:( (2bn1_block2/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_0_block3/kernel
": 2conv2d_0_block3/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn0_block3/gamma
:2bn0_block3/beta
&:$ (2bn0_block3/moving_mean
*:( (2bn0_block3/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_1_block3/kernel
": 2conv2d_1_block3/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn1_block3/gamma
:2bn1_block3/beta
&:$ (2bn1_block3/moving_mean
*:( (2bn1_block3/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.02conv2d_0_block4/kernel
": 02conv2d_0_block4/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:02bn0_block4/gamma
:02bn0_block4/beta
&:$0 (2bn0_block4/moving_mean
*:(0 (2bn0_block4/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.002conv2d_1_block4/kernel
": 02conv2d_1_block4/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:02bn1_block4/gamma
:02bn1_block4/beta
&:$0 (2bn1_block4/moving_mean
*:(0 (2bn1_block4/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'002conv2d_1/kernel
:02conv2d_1/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_1/gamma
(:&02batch_normalization_1/beta
1:/0 (2!batch_normalization_1/moving_mean
5:30 (2%batch_normalization_1/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!02conv1d/kernel
:2conv1d/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?	variables
?metrics
?non_trainable_variables
?regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
p
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9"
trackable_list_wrapper
?
A0
B1
X2
Y3
s4
t5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
]
?
thresholds
?accumulator
?	variables
?	keras_api"
_tf_keras_metric
]
?
thresholds
?accumulator
?	variables
?	keras_api"
_tf_keras_metric
]
?
thresholds
?accumulator
?	variables
?	keras_api"
_tf_keras_metric
]
?
thresholds
?accumulator
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
v
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api"
_tf_keras_metric
v
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"
_tf_keras_metric
?
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"
_tf_keras_metric
?
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6853
@__inference_EdgeNN_layer_call_and_return_conditional_losses_7118
@__inference_EdgeNN_layer_call_and_return_conditional_losses_5889
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6154?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_1194skel_imgnode_pos	node_pair"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_summation_layer_call_and_return_conditional_losses_7128
C__inference_summation_layer_call_and_return_conditional_losses_7138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
(__inference_summation_layer_call_fn_7148
(__inference_summation_layer_call_fn_7158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
I__inference_conv2d_0_block1_layer_call_and_return_conditional_losses_7168?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_0_block1_layer_call_fn_7178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7196
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7214
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7232
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7250?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn0_block1_layer_call_fn_7268
)__inference_bn0_block1_layer_call_fn_7286
)__inference_bn0_block1_layer_call_fn_7304
)__inference_bn0_block1_layer_call_fn_7322?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_concatenate_layer_call_and_return_conditional_losses_7329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_concatenate_layer_call_fn_7336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_relu0_block1_layer_call_and_return_conditional_losses_7341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu0_block1_layer_call_fn_7346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv2d_1_block1_layer_call_and_return_conditional_losses_7356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_1_block1_layer_call_fn_7366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7384
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7402
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7420
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7438?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn1_block1_layer_call_fn_7456
)__inference_bn1_block1_layer_call_fn_7474
)__inference_bn1_block1_layer_call_fn_7492
)__inference_bn1_block1_layer_call_fn_7510?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_concatenate_1_layer_call_and_return_conditional_losses_7517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_1_layer_call_fn_7524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_relu1_block1_layer_call_and_return_conditional_losses_7529?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu1_block1_layer_call_fn_7534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7539
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_max_pooling2d_layer_call_fn_7549
,__inference_max_pooling2d_layer_call_fn_7554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv2d_0_block2_layer_call_and_return_conditional_losses_7564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_0_block2_layer_call_fn_7574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7592
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7610
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7628
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7646?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn0_block2_layer_call_fn_7664
)__inference_bn0_block2_layer_call_fn_7682
)__inference_bn0_block2_layer_call_fn_7700
)__inference_bn0_block2_layer_call_fn_7718?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_relu0_block2_layer_call_and_return_conditional_losses_7723?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu0_block2_layer_call_fn_7728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv2d_1_block2_layer_call_and_return_conditional_losses_7738?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_1_block2_layer_call_fn_7748?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7766
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7784
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7802
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7820?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn1_block2_layer_call_fn_7838
)__inference_bn1_block2_layer_call_fn_7856
)__inference_bn1_block2_layer_call_fn_7874
)__inference_bn1_block2_layer_call_fn_7892?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_relu1_block2_layer_call_and_return_conditional_losses_7897?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu1_block2_layer_call_fn_7902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7907
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7912?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_1_layer_call_fn_7917
.__inference_max_pooling2d_1_layer_call_fn_7922?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv2d_0_block3_layer_call_and_return_conditional_losses_7932?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_0_block3_layer_call_fn_7942?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7960
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7978
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7996
D__inference_bn0_block3_layer_call_and_return_conditional_losses_8014?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn0_block3_layer_call_fn_8032
)__inference_bn0_block3_layer_call_fn_8050
)__inference_bn0_block3_layer_call_fn_8068
)__inference_bn0_block3_layer_call_fn_8086?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_relu0_block3_layer_call_and_return_conditional_losses_8091?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu0_block3_layer_call_fn_8096?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv2d_1_block3_layer_call_and_return_conditional_losses_8106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_1_block3_layer_call_fn_8116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8134
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8152
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8170
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8188?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn1_block3_layer_call_fn_8206
)__inference_bn1_block3_layer_call_fn_8224
)__inference_bn1_block3_layer_call_fn_8242
)__inference_bn1_block3_layer_call_fn_8260?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_relu1_block3_layer_call_and_return_conditional_losses_8265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu1_block3_layer_call_fn_8270?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv2d_layer_call_and_return_conditional_losses_8280?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv2d_layer_call_fn_8290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8308
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8326
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8344
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8362?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_batch_normalization_layer_call_fn_8380
2__inference_batch_normalization_layer_call_fn_8398
2__inference_batch_normalization_layer_call_fn_8416
2__inference_batch_normalization_layer_call_fn_8434?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_relu_C3_block3_layer_call_and_return_conditional_losses_8439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_relu_C3_block3_layer_call_fn_8444?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8449
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_2_layer_call_fn_8459
.__inference_max_pooling2d_2_layer_call_fn_8464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv2d_0_block4_layer_call_and_return_conditional_losses_8474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_0_block4_layer_call_fn_8484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8502
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8520
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8538
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8556?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn0_block4_layer_call_fn_8574
)__inference_bn0_block4_layer_call_fn_8592
)__inference_bn0_block4_layer_call_fn_8610
)__inference_bn0_block4_layer_call_fn_8628?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_relu0_block4_layer_call_and_return_conditional_losses_8633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu0_block4_layer_call_fn_8638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_conv2d_1_block4_layer_call_and_return_conditional_losses_8648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_conv2d_1_block4_layer_call_fn_8658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8676
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8694
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8712
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8730?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_bn1_block4_layer_call_fn_8748
)__inference_bn1_block4_layer_call_fn_8766
)__inference_bn1_block4_layer_call_fn_8784
)__inference_bn1_block4_layer_call_fn_8802?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_relu1_block4_layer_call_and_return_conditional_losses_8807?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_relu1_block4_layer_call_fn_8812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8822?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_1_layer_call_fn_8832?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8850
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8868
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8886
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8904?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_1_layer_call_fn_8922
4__inference_batch_normalization_1_layer_call_fn_8940
4__inference_batch_normalization_1_layer_call_fn_8958
4__inference_batch_normalization_1_layer_call_fn_8976?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_relu_C3_block4_layer_call_and_return_conditional_losses_8981?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_relu_C3_block4_layer_call_fn_8986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8991
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8996?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_3_layer_call_fn_9001
.__inference_max_pooling2d_3_layer_call_fn_9006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9012
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9018?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_global_max_pooling2d_layer_call_fn_9024
3__inference_global_max_pooling2d_layer_call_fn_9030?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv1d_layer_call_and_return_conditional_losses_9068?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv1d_layer_call_fn_9106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_EdgeNN_layer_call_fn_3034
%__inference_EdgeNN_layer_call_fn_9371
%__inference_EdgeNN_layer_call_fn_9636
%__inference_EdgeNN_layer_call_fn_5624?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference_signature_wrapper_6588	node_pairnode_posskel_img"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
@__inference_EdgeNN_layer_call_and_return_conditional_losses_5889?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
skel_img???????????
,?)
node_pos???????????
-?*
	node_pair???????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6154?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
skel_img???????????
,?)
node_pos???????????
-?*
	node_pair???????????
p

 
? "%?"
?
0?????????
? ?
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6853?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_EdgeNN_layer_call_and_return_conditional_losses_7118?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
p

 
? "%?"
?
0?????????
? ?
%__inference_EdgeNN_layer_call_fn_3034?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
skel_img???????????
,?)
node_pos???????????
-?*
	node_pair???????????
p 

 
? "???????????
%__inference_EdgeNN_layer_call_fn_5624?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
skel_img???????????
,?)
node_pos???????????
-?*
	node_pair???????????
p

 
? "???????????
%__inference_EdgeNN_layer_call_fn_9371?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
p 

 
? "???????????
%__inference_EdgeNN_layer_call_fn_9636?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
p

 
? "???????????
__inference__wrapped_model_1194?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
???
???
,?)
skel_img???????????
,?)
node_pos???????????
-?*
	node_pair???????????
? "K?H
F
tf.compat.v1.squeeze.?+
tf.compat.v1.squeeze??????????
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8850?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8868?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8886v????;?8
1?.
(?%
inputs?????????  0
p 
? "-?*
#? 
0?????????  0
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8904v????;?8
1?.
(?%
inputs?????????  0
p
? "-?*
#? 
0?????????  0
? ?
4__inference_batch_normalization_1_layer_call_fn_8922?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
4__inference_batch_normalization_1_layer_call_fn_8940?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
4__inference_batch_normalization_1_layer_call_fn_8958i????;?8
1?.
(?%
inputs?????????  0
p 
? " ??????????  0?
4__inference_batch_normalization_1_layer_call_fn_8976i????;?8
1?.
(?%
inputs?????????  0
p
? " ??????????  0?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8308?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8326?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8344v????;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8362v????;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
2__inference_batch_normalization_layer_call_fn_8380?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
2__inference_batch_normalization_layer_call_fn_8398?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
2__inference_batch_normalization_layer_call_fn_8416i????;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
2__inference_batch_normalization_layer_call_fn_8434i????;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7196??@ABM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7214??@ABM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7232v?@AB=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7250v?@AB=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
)__inference_bn0_block1_layer_call_fn_7268??@ABM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
)__inference_bn0_block1_layer_call_fn_7286??@ABM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
)__inference_bn0_block1_layer_call_fn_7304i?@AB=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
)__inference_bn0_block1_layer_call_fn_7322i?@AB=?:
3?0
*?'
inputs???????????
p
? ""?????????????
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7592?qrstM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7610?qrstM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7628vqrst=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7646vqrst=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
)__inference_bn0_block2_layer_call_fn_7664?qrstM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
)__inference_bn0_block2_layer_call_fn_7682?qrstM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
)__inference_bn0_block2_layer_call_fn_7700iqrst=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
)__inference_bn0_block2_layer_call_fn_7718iqrst=?:
3?0
*?'
inputs???????????
p
? ""?????????????
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7960?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7978?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7996v????;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
D__inference_bn0_block3_layer_call_and_return_conditional_losses_8014v????;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
)__inference_bn0_block3_layer_call_fn_8032?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
)__inference_bn0_block3_layer_call_fn_8050?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
)__inference_bn0_block3_layer_call_fn_8068i????;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
)__inference_bn0_block3_layer_call_fn_8086i????;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8502?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8520?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8538v????;?8
1?.
(?%
inputs?????????  0
p 
? "-?*
#? 
0?????????  0
? ?
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8556v????;?8
1?.
(?%
inputs?????????  0
p
? "-?*
#? 
0?????????  0
? ?
)__inference_bn0_block4_layer_call_fn_8574?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
)__inference_bn0_block4_layer_call_fn_8592?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
)__inference_bn0_block4_layer_call_fn_8610i????;?8
1?.
(?%
inputs?????????  0
p 
? " ??????????  0?
)__inference_bn0_block4_layer_call_fn_8628i????;?8
1?.
(?%
inputs?????????  0
p
? " ??????????  0?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7384?VWXYM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7402?VWXYM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7420vVWXY=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7438vVWXY=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
)__inference_bn1_block1_layer_call_fn_7456?VWXYM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
)__inference_bn1_block1_layer_call_fn_7474?VWXYM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
)__inference_bn1_block1_layer_call_fn_7492iVWXY=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
)__inference_bn1_block1_layer_call_fn_7510iVWXY=?:
3?0
*?'
inputs???????????
p
? ""?????????????
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7766?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7784?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7802z????=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7820z????=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
)__inference_bn1_block2_layer_call_fn_7838?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
)__inference_bn1_block2_layer_call_fn_7856?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
)__inference_bn1_block2_layer_call_fn_7874m????=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
)__inference_bn1_block2_layer_call_fn_7892m????=?:
3?0
*?'
inputs???????????
p
? ""?????????????
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8134?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8152?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8170v????;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8188v????;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
)__inference_bn1_block3_layer_call_fn_8206?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
)__inference_bn1_block3_layer_call_fn_8224?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
)__inference_bn1_block3_layer_call_fn_8242i????;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
)__inference_bn1_block3_layer_call_fn_8260i????;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8676?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8694?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8712v????;?8
1?.
(?%
inputs?????????  0
p 
? "-?*
#? 
0?????????  0
? ?
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8730v????;?8
1?.
(?%
inputs?????????  0
p
? "-?*
#? 
0?????????  0
? ?
)__inference_bn1_block4_layer_call_fn_8748?????M?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
)__inference_bn1_block4_layer_call_fn_8766?????M?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
)__inference_bn1_block4_layer_call_fn_8784i????;?8
1?.
(?%
inputs?????????  0
p 
? " ??????????  0?
)__inference_bn1_block4_layer_call_fn_8802i????;?8
1?.
(?%
inputs?????????  0
p
? " ??????????  0?
G__inference_concatenate_1_layer_call_and_return_conditional_losses_7517?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? "/?,
%?"
0???????????
? ?
,__inference_concatenate_1_layer_call_fn_7524?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""?????????????
E__inference_concatenate_layer_call_and_return_conditional_losses_7329?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? "/?,
%?"
0???????????
? ?
*__inference_concatenate_layer_call_fn_7336?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""?????????????
@__inference_conv1d_layer_call_and_return_conditional_losses_9068n??7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????
? ?
%__inference_conv1d_layer_call_fn_9106a??7?4
-?*
(?%
inputs?????????0
? " ???????????
I__inference_conv2d_0_block1_layer_call_and_return_conditional_losses_7168p899?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_conv2d_0_block1_layer_call_fn_7178c899?6
/?,
*?'
inputs???????????
? ""?????????????
I__inference_conv2d_0_block2_layer_call_and_return_conditional_losses_7564pjk9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_conv2d_0_block2_layer_call_fn_7574cjk9?6
/?,
*?'
inputs???????????
? ""?????????????
I__inference_conv2d_0_block3_layer_call_and_return_conditional_losses_7932n??7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
.__inference_conv2d_0_block3_layer_call_fn_7942a??7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
I__inference_conv2d_0_block4_layer_call_and_return_conditional_losses_8474n??7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  0
? ?
.__inference_conv2d_0_block4_layer_call_fn_8484a??7?4
-?*
(?%
inputs?????????  
? " ??????????  0?
I__inference_conv2d_1_block1_layer_call_and_return_conditional_losses_7356pOP9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_conv2d_1_block1_layer_call_fn_7366cOP9?6
/?,
*?'
inputs???????????
? ""?????????????
I__inference_conv2d_1_block2_layer_call_and_return_conditional_losses_7738p}~9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_conv2d_1_block2_layer_call_fn_7748c}~9?6
/?,
*?'
inputs???????????
? ""?????????????
I__inference_conv2d_1_block3_layer_call_and_return_conditional_losses_8106n??7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
.__inference_conv2d_1_block3_layer_call_fn_8116a??7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
I__inference_conv2d_1_block4_layer_call_and_return_conditional_losses_8648n??7?4
-?*
(?%
inputs?????????  0
? "-?*
#? 
0?????????  0
? ?
.__inference_conv2d_1_block4_layer_call_fn_8658a??7?4
-?*
(?%
inputs?????????  0
? " ??????????  0?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8822n??7?4
-?*
(?%
inputs?????????  0
? "-?*
#? 
0?????????  0
? ?
'__inference_conv2d_1_layer_call_fn_8832a??7?4
-?*
(?%
inputs?????????  0
? " ??????????  0?
@__inference_conv2d_layer_call_and_return_conditional_losses_8280n??7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
%__inference_conv2d_layer_call_fn_8290a??7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9012?R?O
H?E
C?@
inputs4????????????????????????????????????
? "6?3
,?)
0"??????????????????
? ?
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9018h7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????0
? ?
3__inference_global_max_pooling2d_layer_call_fn_9024R?O
H?E
C?@
inputs4????????????????????????????????????
? ")?&"???????????????????
3__inference_global_max_pooling2d_layer_call_fn_9030[7?4
-?*
(?%
inputs?????????0
? " ??????????0?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7907?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7912j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@
? ?
.__inference_max_pooling2d_1_layer_call_fn_7917?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_max_pooling2d_1_layer_call_fn_7922]9?6
/?,
*?'
inputs???????????
? " ??????????@@?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8449?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8454h7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????  
? ?
.__inference_max_pooling2d_2_layer_call_fn_8459?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_max_pooling2d_2_layer_call_fn_8464[7?4
-?*
(?%
inputs?????????@@
? " ??????????  ?
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8991?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8996h7?4
-?*
(?%
inputs?????????  0
? "-?*
#? 
0?????????0
? ?
.__inference_max_pooling2d_3_layer_call_fn_9001?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_max_pooling2d_3_layer_call_fn_9006[7?4
-?*
(?%
inputs?????????  0
? " ??????????0?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7539?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7544l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_max_pooling2d_layer_call_fn_7549?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
,__inference_max_pooling2d_layer_call_fn_7554_9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_relu0_block1_layer_call_and_return_conditional_losses_7341l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_relu0_block1_layer_call_fn_7346_9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_relu0_block2_layer_call_and_return_conditional_losses_7723l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_relu0_block2_layer_call_fn_7728_9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_relu0_block3_layer_call_and_return_conditional_losses_8091h7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
+__inference_relu0_block3_layer_call_fn_8096[7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
F__inference_relu0_block4_layer_call_and_return_conditional_losses_8633h7?4
-?*
(?%
inputs?????????  0
? "-?*
#? 
0?????????  0
? ?
+__inference_relu0_block4_layer_call_fn_8638[7?4
-?*
(?%
inputs?????????  0
? " ??????????  0?
F__inference_relu1_block1_layer_call_and_return_conditional_losses_7529l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_relu1_block1_layer_call_fn_7534_9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_relu1_block2_layer_call_and_return_conditional_losses_7897l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_relu1_block2_layer_call_fn_7902_9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_relu1_block3_layer_call_and_return_conditional_losses_8265h7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
+__inference_relu1_block3_layer_call_fn_8270[7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
F__inference_relu1_block4_layer_call_and_return_conditional_losses_8807h7?4
-?*
(?%
inputs?????????  0
? "-?*
#? 
0?????????  0
? ?
+__inference_relu1_block4_layer_call_fn_8812[7?4
-?*
(?%
inputs?????????  0
? " ??????????  0?
H__inference_relu_C3_block3_layer_call_and_return_conditional_losses_8439h7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
-__inference_relu_C3_block3_layer_call_fn_8444[7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
H__inference_relu_C3_block4_layer_call_and_return_conditional_losses_8981h7?4
-?*
(?%
inputs?????????  0
? "-?*
#? 
0?????????  0
? ?
-__inference_relu_C3_block4_layer_call_fn_8986[7?4
-?*
(?%
inputs?????????  0
? " ??????????  0?
"__inference_signature_wrapper_6588?h89?@ABOPVWXYjkqrst}~?????????????????????????????????????????????
? 
???
:
	node_pair-?*
	node_pair???????????
8
node_pos,?)
node_pos???????????
8
skel_img,?)
skel_img???????????"K?H
F
tf.compat.v1.squeeze.?+
tf.compat.v1.squeeze??????????
C__inference_summation_layer_call_and_return_conditional_losses_7128????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
?

trainingp "/?,
%?"
0???????????
? ?
C__inference_summation_layer_call_and_return_conditional_losses_7138????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
?

trainingp"/?,
%?"
0???????????
? ?
(__inference_summation_layer_call_fn_7148????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
?

trainingp ""?????????????
(__inference_summation_layer_call_fn_7158????
???
???
,?)
inputs/0???????????
,?)
inputs/1???????????
,?)
inputs/2???????????
?

trainingp""????????????