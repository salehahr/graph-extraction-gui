вЉ6
зЛ
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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
њ
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
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.22unknown8з0

conv2d_0_block1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_0_block1/kernel

*conv2d_0_block1/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block1/kernel*&
_output_shapes
:*
dtype0

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

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

bn0_block1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn0_block1/moving_variance

.bn0_block1/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block1/moving_variance*
_output_shapes
:*
dtype0

conv2d_1_block1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_1_block1/kernel

*conv2d_1_block1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block1/kernel*&
_output_shapes
:*
dtype0

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

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

bn1_block1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn1_block1/moving_variance

.bn1_block1/moving_variance/Read/ReadVariableOpReadVariableOpbn1_block1/moving_variance*
_output_shapes
:*
dtype0

conv2d_0_block2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_0_block2/kernel

*conv2d_0_block2/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block2/kernel*&
_output_shapes
:*
dtype0

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

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

bn0_block2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn0_block2/moving_variance

.bn0_block2/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block2/moving_variance*
_output_shapes
:*
dtype0

conv2d_1_block2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_1_block2/kernel

*conv2d_1_block2/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block2/kernel*&
_output_shapes
:*
dtype0

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

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

bn1_block2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn1_block2/moving_variance

.bn1_block2/moving_variance/Read/ReadVariableOpReadVariableOpbn1_block2/moving_variance*
_output_shapes
:*
dtype0

conv2d_0_block3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_0_block3/kernel

*conv2d_0_block3/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block3/kernel*&
_output_shapes
:*
dtype0

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

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

bn0_block3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn0_block3/moving_variance

.bn0_block3/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block3/moving_variance*
_output_shapes
:*
dtype0

conv2d_1_block3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv2d_1_block3/kernel

*conv2d_1_block3/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block3/kernel*&
_output_shapes
:*
dtype0

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

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

bn1_block3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebn1_block3/moving_variance

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

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0

conv2d_0_block4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameconv2d_0_block4/kernel

*conv2d_0_block4/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_block4/kernel*&
_output_shapes
:0*
dtype0

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

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

bn0_block4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebn0_block4/moving_variance

.bn0_block4/moving_variance/Read/ReadVariableOpReadVariableOpbn0_block4/moving_variance*
_output_shapes
:0*
dtype0

conv2d_1_block4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameconv2d_1_block4/kernel

*conv2d_1_block4/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_block4/kernel*&
_output_shapes
:00*
dtype0

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

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

bn1_block4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebn1_block4/moving_variance

.bn1_block4/moving_variance/Read/ReadVariableOpReadVariableOpbn1_block4/moving_variance*
_output_shapes
:0*
dtype0

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

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:0*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:0*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:0*
dtype0
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_1/moving_variance

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
shape:Ш*!
shared_nametrue_positives_2
r
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:Ш*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:Ш*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:Ш*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:Ш*
dtype0
y
true_positives_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*!
shared_nametrue_positives_3
r
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes	
:Ш*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:Ш*
dtype0
{
false_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*"
shared_namefalse_positives_2
t
%false_positives_2/Read/ReadVariableOpReadVariableOpfalse_positives_2*
_output_shapes	
:Ш*
dtype0
{
false_negatives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*"
shared_namefalse_negatives_2
t
%false_negatives_2/Read/ReadVariableOpReadVariableOpfalse_negatives_2*
_output_shapes	
:Ш*
dtype0

NoOpNoOp
бЧ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ч
valueЧBќЦ BєЦ


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

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

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

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
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
V
Ѓ	variables
Єregularization_losses
Ѕtrainable_variables
І	keras_api
n
Їkernel
	Јbias
Љ	variables
Њregularization_losses
Ћtrainable_variables
Ќ	keras_api
 
	­axis

Ўgamma
	Џbeta
Аmoving_mean
Бmoving_variance
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
V
Ж	variables
Зregularization_losses
Иtrainable_variables
Й	keras_api
n
Кkernel
	Лbias
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
 
	Рaxis

Сgamma
	Тbeta
Уmoving_mean
Фmoving_variance
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api
V
Щ	variables
Ъregularization_losses
Ыtrainable_variables
Ь	keras_api
V
Э	variables
Юregularization_losses
Яtrainable_variables
а	keras_api
n
бkernel
	вbias
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
 
	зaxis

иgamma
	йbeta
кmoving_mean
лmoving_variance
м	variables
нregularization_losses
оtrainable_variables
п	keras_api
V
р	variables
сregularization_losses
тtrainable_variables
у	keras_api
n
фkernel
	хbias
ц	variables
чregularization_losses
шtrainable_variables
щ	keras_api
 
	ъaxis

ыgamma
	ьbeta
эmoving_mean
юmoving_variance
я	variables
№regularization_losses
ёtrainable_variables
ђ	keras_api
V
ѓ	variables
єregularization_losses
ѕtrainable_variables
і	keras_api
n
їkernel
	јbias
љ	variables
њregularization_losses
ћtrainable_variables
ќ	keras_api
 
	§axis

ўgamma
	џbeta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api

	keras_api
 
 
 

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
20
21
22
23
24
25
26
27
28
29
Ї30
Ј31
Ў32
Џ33
А34
Б35
К36
Л37
С38
Т39
У40
Ф41
б42
в43
и44
й45
к46
л47
ф48
х49
ы50
ь51
э52
ю53
ї54
ј55
ў56
џ57
58
59
60
61
 
т
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
14
15
16
17
18
19
Ї20
Ј21
Ў22
Џ23
К24
Л25
С26
Т27
б28
в29
и30
й31
ф32
х33
ы34
ь35
ї36
ј37
ў38
џ39
40
41
В
 layer_regularization_losses
layer_metrics
/	variables
metrics
non_trainable_variables
0regularization_losses
layers
1trainable_variables
 
 
 
 
В
 layer_regularization_losses
layer_metrics
4	variables
 metrics
Ёnon_trainable_variables
5regularization_losses
Ђlayers
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
В
 Ѓlayer_regularization_losses
Єlayer_metrics
:	variables
Ѕmetrics
Іnon_trainable_variables
;regularization_losses
Їlayers
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
В
 Јlayer_regularization_losses
Љlayer_metrics
C	variables
Њmetrics
Ћnon_trainable_variables
Dregularization_losses
Ќlayers
Etrainable_variables
 
 
 
В
 ­layer_regularization_losses
Ўlayer_metrics
G	variables
Џmetrics
Аnon_trainable_variables
Hregularization_losses
Бlayers
Itrainable_variables
 
 
 
В
 Вlayer_regularization_losses
Гlayer_metrics
K	variables
Дmetrics
Еnon_trainable_variables
Lregularization_losses
Жlayers
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
В
 Зlayer_regularization_losses
Иlayer_metrics
Q	variables
Йmetrics
Кnon_trainable_variables
Rregularization_losses
Лlayers
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
В
 Мlayer_regularization_losses
Нlayer_metrics
Z	variables
Оmetrics
Пnon_trainable_variables
[regularization_losses
Рlayers
\trainable_variables
 
 
 
В
 Сlayer_regularization_losses
Тlayer_metrics
^	variables
Уmetrics
Фnon_trainable_variables
_regularization_losses
Хlayers
`trainable_variables
 
 
 
В
 Цlayer_regularization_losses
Чlayer_metrics
b	variables
Шmetrics
Щnon_trainable_variables
cregularization_losses
Ъlayers
dtrainable_variables
 
 
 
В
 Ыlayer_regularization_losses
Ьlayer_metrics
f	variables
Эmetrics
Юnon_trainable_variables
gregularization_losses
Яlayers
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
В
 аlayer_regularization_losses
бlayer_metrics
l	variables
вmetrics
гnon_trainable_variables
mregularization_losses
дlayers
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
В
 еlayer_regularization_losses
жlayer_metrics
u	variables
зmetrics
иnon_trainable_variables
vregularization_losses
йlayers
wtrainable_variables
 
 
 
В
 кlayer_regularization_losses
лlayer_metrics
y	variables
мmetrics
нnon_trainable_variables
zregularization_losses
оlayers
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
Д
 пlayer_regularization_losses
рlayer_metrics
	variables
сmetrics
тnon_trainable_variables
regularization_losses
уlayers
trainable_variables
 
[Y
VARIABLE_VALUEbn1_block2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbn1_block2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbn1_block2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn1_block2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
Е
 фlayer_regularization_losses
хlayer_metrics
	variables
цmetrics
чnon_trainable_variables
regularization_losses
шlayers
trainable_variables
 
 
 
Е
 щlayer_regularization_losses
ъlayer_metrics
	variables
ыmetrics
ьnon_trainable_variables
regularization_losses
эlayers
trainable_variables
 
 
 
Е
 юlayer_regularization_losses
яlayer_metrics
	variables
№metrics
ёnon_trainable_variables
regularization_losses
ђlayers
trainable_variables
b`
VARIABLE_VALUEconv2d_0_block3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_0_block3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Е
 ѓlayer_regularization_losses
єlayer_metrics
	variables
ѕmetrics
іnon_trainable_variables
regularization_losses
їlayers
trainable_variables
 
[Y
VARIABLE_VALUEbn0_block3/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbn0_block3/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbn0_block3/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn0_block3/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
Е
 јlayer_regularization_losses
љlayer_metrics
	variables
њmetrics
ћnon_trainable_variables
 regularization_losses
ќlayers
Ёtrainable_variables
 
 
 
Е
 §layer_regularization_losses
ўlayer_metrics
Ѓ	variables
џmetrics
non_trainable_variables
Єregularization_losses
layers
Ѕtrainable_variables
ca
VARIABLE_VALUEconv2d_1_block3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_1_block3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Ї0
Ј1
 

Ї0
Ј1
Е
 layer_regularization_losses
layer_metrics
Љ	variables
metrics
non_trainable_variables
Њregularization_losses
layers
Ћtrainable_variables
 
\Z
VARIABLE_VALUEbn1_block3/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbn1_block3/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn1_block3/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbn1_block3/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ў0
Џ1
А2
Б3
 

Ў0
Џ1
Е
 layer_regularization_losses
layer_metrics
В	variables
metrics
non_trainable_variables
Гregularization_losses
layers
Дtrainable_variables
 
 
 
Е
 layer_regularization_losses
layer_metrics
Ж	variables
metrics
non_trainable_variables
Зregularization_losses
layers
Иtrainable_variables
ZX
VARIABLE_VALUEconv2d/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

К0
Л1
 

К0
Л1
Е
 layer_regularization_losses
layer_metrics
М	variables
metrics
non_trainable_variables
Нregularization_losses
layers
Оtrainable_variables
 
ec
VARIABLE_VALUEbatch_normalization/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEbatch_normalization/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#batch_normalization/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
С0
Т1
У2
Ф3
 

С0
Т1
Е
 layer_regularization_losses
layer_metrics
Х	variables
metrics
non_trainable_variables
Цregularization_losses
layers
Чtrainable_variables
 
 
 
Е
 layer_regularization_losses
layer_metrics
Щ	variables
metrics
non_trainable_variables
Ъregularization_losses
layers
Ыtrainable_variables
 
 
 
Е
  layer_regularization_losses
Ёlayer_metrics
Э	variables
Ђmetrics
Ѓnon_trainable_variables
Юregularization_losses
Єlayers
Яtrainable_variables
ca
VARIABLE_VALUEconv2d_0_block4/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_0_block4/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

б0
в1
 

б0
в1
Е
 Ѕlayer_regularization_losses
Іlayer_metrics
г	variables
Їmetrics
Јnon_trainable_variables
дregularization_losses
Љlayers
еtrainable_variables
 
\Z
VARIABLE_VALUEbn0_block4/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbn0_block4/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn0_block4/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbn0_block4/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
и0
й1
к2
л3
 

и0
й1
Е
 Њlayer_regularization_losses
Ћlayer_metrics
м	variables
Ќmetrics
­non_trainable_variables
нregularization_losses
Ўlayers
оtrainable_variables
 
 
 
Е
 Џlayer_regularization_losses
Аlayer_metrics
р	variables
Бmetrics
Вnon_trainable_variables
сregularization_losses
Гlayers
тtrainable_variables
ca
VARIABLE_VALUEconv2d_1_block4/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_1_block4/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

ф0
х1
 

ф0
х1
Е
 Дlayer_regularization_losses
Еlayer_metrics
ц	variables
Жmetrics
Зnon_trainable_variables
чregularization_losses
Иlayers
шtrainable_variables
 
\Z
VARIABLE_VALUEbn1_block4/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbn1_block4/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn1_block4/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbn1_block4/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
ы0
ь1
э2
ю3
 

ы0
ь1
Е
 Йlayer_regularization_losses
Кlayer_metrics
я	variables
Лmetrics
Мnon_trainable_variables
№regularization_losses
Нlayers
ёtrainable_variables
 
 
 
Е
 Оlayer_regularization_losses
Пlayer_metrics
ѓ	variables
Рmetrics
Сnon_trainable_variables
єregularization_losses
Тlayers
ѕtrainable_variables
\Z
VARIABLE_VALUEconv2d_1/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_1/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

ї0
ј1
 

ї0
ј1
Е
 Уlayer_regularization_losses
Фlayer_metrics
љ	variables
Хmetrics
Цnon_trainable_variables
њregularization_losses
Чlayers
ћtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_1/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_1/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_1/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_1/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
ў0
џ1
2
3
 

ў0
џ1
Е
 Шlayer_regularization_losses
Щlayer_metrics
	variables
Ъmetrics
Ыnon_trainable_variables
regularization_losses
Ьlayers
trainable_variables
 
 
 
Е
 Эlayer_regularization_losses
Юlayer_metrics
	variables
Яmetrics
аnon_trainable_variables
regularization_losses
бlayers
trainable_variables
 
 
 
Е
 вlayer_regularization_losses
гlayer_metrics
	variables
дmetrics
еnon_trainable_variables
regularization_losses
жlayers
trainable_variables
 
 
 
Е
 зlayer_regularization_losses
иlayer_metrics
	variables
йmetrics
кnon_trainable_variables
regularization_losses
лlayers
trainable_variables
ZX
VARIABLE_VALUEconv1d/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1d/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Е
 мlayer_regularization_losses
нlayer_metrics
	variables
оmetrics
пnon_trainable_variables
regularization_losses
рlayers
trainable_variables
 
 
 
P
с0
т1
у2
ф3
х4
ц5
ч6
ш7
щ8
ъ9
Є
A0
B1
X2
Y3
s4
t5
6
7
8
9
А10
Б11
У12
Ф13
к14
л15
э16
ю17
18
19
Ю
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
0
1
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
0
1
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
А0
Б1
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
У0
Ф1
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
к0
л1
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
э0
ю1
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
0
1
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

ыtotal

ьcount
э	variables
ю	keras_api
C
я
thresholds
№accumulator
ё	variables
ђ	keras_api
C
ѓ
thresholds
єaccumulator
ѕ	variables
і	keras_api
C
ї
thresholds
јaccumulator
љ	variables
њ	keras_api
C
ћ
thresholds
ќaccumulator
§	variables
ў	keras_api
I

џtotal

count

_fn_kwargs
	variables
	keras_api
\

thresholds
true_positives
false_positives
	variables
	keras_api
\

thresholds
true_positives
false_negatives
	variables
	keras_api
v
true_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api
v
true_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ы0
ь1

э	variables
 
[Y
VARIABLE_VALUEaccumulator:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUE

№0

ё	variables
 
][
VARIABLE_VALUEaccumulator_1:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUE

є0

ѕ	variables
 
][
VARIABLE_VALUEaccumulator_2:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUE

ј0

љ	variables
 
][
VARIABLE_VALUEaccumulator_3:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUE

ќ0

§	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

џ0
1

	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/8/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/8/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3

	variables
ca
VARIABLE_VALUEtrue_positives_3=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/9/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_2>keras_api/metrics/9/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_2>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3

	variables

serving_default_node_pairPlaceholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ

serving_default_node_posPlaceholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ

serving_default_skel_imgPlaceholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ
х
StatefulPartitionedCallStatefulPartitionedCallserving_default_node_pairserving_default_node_posserving_default_skel_imgconv2d_0_block1/kernelconv2d_0_block1/biasbn0_block1/gammabn0_block1/betabn0_block1/moving_meanbn0_block1/moving_varianceconv2d_1_block1/kernelconv2d_1_block1/biasbn1_block1/gammabn1_block1/betabn1_block1/moving_meanbn1_block1/moving_varianceconv2d_0_block2/kernelconv2d_0_block2/biasbn0_block2/gammabn0_block2/betabn0_block2/moving_meanbn0_block2/moving_varianceconv2d_1_block2/kernelconv2d_1_block2/biasbn1_block2/gammabn1_block2/betabn1_block2/moving_meanbn1_block2/moving_varianceconv2d_0_block3/kernelconv2d_0_block3/biasbn0_block3/gammabn0_block3/betabn0_block3/moving_meanbn0_block3/moving_varianceconv2d_1_block3/kernelconv2d_1_block3/biasbn1_block3/gammabn1_block3/betabn1_block3/moving_meanbn1_block3/moving_varianceconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_0_block4/kernelconv2d_0_block4/biasbn0_block4/gammabn0_block4/betabn0_block4/moving_meanbn0_block4/moving_varianceconv2d_1_block4/kernelconv2d_1_block4/biasbn1_block4/gammabn1_block4/betabn1_block4/moving_meanbn1_block4/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv1d/kernelconv1d/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8 *+
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
д&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*§%
valueѓ%B№%SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH

SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*Л
valueБBЎSB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
о
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices*conv2d_0_block1/kernel/Read/ReadVariableOp(conv2d_0_block1/bias/Read/ReadVariableOp$bn0_block1/gamma/Read/ReadVariableOp#bn0_block1/beta/Read/ReadVariableOp*bn0_block1/moving_mean/Read/ReadVariableOp.bn0_block1/moving_variance/Read/ReadVariableOp*conv2d_1_block1/kernel/Read/ReadVariableOp(conv2d_1_block1/bias/Read/ReadVariableOp$bn1_block1/gamma/Read/ReadVariableOp#bn1_block1/beta/Read/ReadVariableOp*bn1_block1/moving_mean/Read/ReadVariableOp.bn1_block1/moving_variance/Read/ReadVariableOp*conv2d_0_block2/kernel/Read/ReadVariableOp(conv2d_0_block2/bias/Read/ReadVariableOp$bn0_block2/gamma/Read/ReadVariableOp#bn0_block2/beta/Read/ReadVariableOp*bn0_block2/moving_mean/Read/ReadVariableOp.bn0_block2/moving_variance/Read/ReadVariableOp*conv2d_1_block2/kernel/Read/ReadVariableOp(conv2d_1_block2/bias/Read/ReadVariableOp$bn1_block2/gamma/Read/ReadVariableOp#bn1_block2/beta/Read/ReadVariableOp*bn1_block2/moving_mean/Read/ReadVariableOp.bn1_block2/moving_variance/Read/ReadVariableOp*conv2d_0_block3/kernel/Read/ReadVariableOp(conv2d_0_block3/bias/Read/ReadVariableOp$bn0_block3/gamma/Read/ReadVariableOp#bn0_block3/beta/Read/ReadVariableOp*bn0_block3/moving_mean/Read/ReadVariableOp.bn0_block3/moving_variance/Read/ReadVariableOp*conv2d_1_block3/kernel/Read/ReadVariableOp(conv2d_1_block3/bias/Read/ReadVariableOp$bn1_block3/gamma/Read/ReadVariableOp#bn1_block3/beta/Read/ReadVariableOp*bn1_block3/moving_mean/Read/ReadVariableOp.bn1_block3/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp*conv2d_0_block4/kernel/Read/ReadVariableOp(conv2d_0_block4/bias/Read/ReadVariableOp$bn0_block4/gamma/Read/ReadVariableOp#bn0_block4/beta/Read/ReadVariableOp*bn0_block4/moving_mean/Read/ReadVariableOp.bn0_block4/moving_variance/Read/ReadVariableOp*conv2d_1_block4/kernel/Read/ReadVariableOp(conv2d_1_block4/bias/Read/ReadVariableOp$bn1_block4/gamma/Read/ReadVariableOp#bn1_block4/beta/Read/ReadVariableOp*bn1_block4/moving_mean/Read/ReadVariableOp.bn1_block4/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpaccumulator/Read/ReadVariableOp!accumulator_1/Read/ReadVariableOp!accumulator_2/Read/ReadVariableOp!accumulator_3/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp$true_positives_3/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_2/Read/ReadVariableOp%false_negatives_2/Read/ReadVariableOpConst"/device:CPU:0*a
dtypesW
U2S

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
з&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*§%
valueѓ%B№%SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH

RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*Л
valueБBЎSB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Б
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*т
_output_shapesЯ
Ь:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
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
р
Identity_83Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: в+

љ
@__inference_conv2d_layer_call_and_return_conditional_losses_8280

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ј	
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
џџџџџџџџџ2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Sum/reduction_indices
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Щ
Г
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7820

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№

ч
.__inference_conv2d_1_block3_layer_call_fn_8116

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ио
с2
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
identityЂ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisД
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЧ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisЭ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block2/FusedBatchNormV3
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block2/FusedBatchNormV3
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1І
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn0_block3/FusedBatchNormV3
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1І
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn1_block3/FusedBatchNormV3
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1г
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1І
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn0_block4/FusedBatchNormV3
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1І
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn1_block4/FusedBatchNormV3
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityу
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
:џџџџџџџџџ
"
_user_specified_name
skel_img:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
node_pos:\X
1
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	node_pair
щ

р
'__inference_conv2d_1_layer_call_fn_8832

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs

b
F__inference_relu0_block2_layer_call_and_return_conditional_losses_7723

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с

O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8850

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
Ў

)__inference_bn1_block2_layer_call_fn_7892

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


I__inference_conv2d_0_block1_layer_call_and_return_conditional_losses_7168

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
Ѓ
4__inference_batch_normalization_1_layer_call_fn_8940

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
№

ч
.__inference_conv2d_0_block3_layer_call_fn_7942

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
№

ч
.__inference_conv2d_0_block4_layer_call_fn_8484

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
№

ч
.__inference_conv2d_1_block4_layer_call_fn_8658

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs

Г
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8152

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Л
є
)__inference_bn0_block2_layer_call_fn_7664

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
С
Г
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8188

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

J
.__inference_max_pooling2d_2_layer_call_fn_8459

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Г
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7610

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў

)__inference_bn0_block1_layer_call_fn_7322

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ
ш7
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
identityЂ"batch_normalization/AssignNewValueЂ$batch_normalization/AssignNewValue_1Ђ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ$batch_normalization_1/AssignNewValueЂ&batch_normalization_1/AssignNewValue_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђbn0_block1/AssignNewValueЂbn0_block1/AssignNewValue_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђbn0_block2/AssignNewValueЂbn0_block2/AssignNewValue_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђbn0_block3/AssignNewValueЂbn0_block3/AssignNewValue_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђbn0_block4/AssignNewValueЂbn0_block4/AssignNewValue_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђbn1_block1/AssignNewValueЂbn1_block1/AssignNewValue_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђbn1_block2/AssignNewValueЂbn1_block2/AssignNewValue_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђbn1_block3/AssignNewValueЂbn1_block3/AssignNewValue_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђbn1_block4/AssignNewValueЂbn1_block4/AssignNewValue_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisД
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block1/FusedBatchNormV3љ
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue
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
concatenate/concat/axisЧ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block1/FusedBatchNormV3љ
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue
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
concatenate_1/concat/axisЭ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block2/FusedBatchNormV3љ
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block2/FusedBatchNormV3љ
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block3/FusedBatchNormV3љ
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block3/FusedBatchNormV3љ
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1с
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2&
$batch_normalization/FusedBatchNormV3І
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueВ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block4/FusedBatchNormV3љ
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block4/FusedBatchNormV3љ
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1я
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2(
&batch_normalization_1/FusedBatchNormV3А
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueМ
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЯ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:џџџџџџџџџ
"
_user_specified_name
skel_img:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
node_pos:\X
1
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	node_pair
М
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8454

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
у
G
+__inference_relu0_block4_layer_call_fn_8638

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs

d
H__inference_relu_C3_block4_layer_call_and_return_conditional_losses_8981

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
Щ
Г
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7438

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
н
O
3__inference_global_max_pooling2d_layer_call_fn_9024

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices
MaxMaxinputsMax/reduction_indices:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
	keep_dims(2
Maxq
IdentityIdentityMax:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

П9
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
identityЂ:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOpЂ<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ)EdgeNN/batch_normalization/ReadVariableOpЂ+EdgeNN/batch_normalization/ReadVariableOp_1Ђ<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ+EdgeNN/batch_normalization_1/ReadVariableOpЂ-EdgeNN/batch_normalization_1/ReadVariableOp_1Ђ1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn0_block1/ReadVariableOpЂ"EdgeNN/bn0_block1/ReadVariableOp_1Ђ1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn0_block2/ReadVariableOpЂ"EdgeNN/bn0_block2/ReadVariableOp_1Ђ1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn0_block3/ReadVariableOpЂ"EdgeNN/bn0_block3/ReadVariableOp_1Ђ1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn0_block4/ReadVariableOpЂ"EdgeNN/bn0_block4/ReadVariableOp_1Ђ1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn1_block1/ReadVariableOpЂ"EdgeNN/bn1_block1/ReadVariableOp_1Ђ1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn1_block2/ReadVariableOpЂ"EdgeNN/bn1_block2/ReadVariableOp_1Ђ1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn1_block3/ReadVariableOpЂ"EdgeNN/bn1_block3/ReadVariableOp_1Ђ1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOpЂ3EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђ EdgeNN/bn1_block4/ReadVariableOpЂ"EdgeNN/bn1_block4/ReadVariableOp_1Ђ0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂ$EdgeNN/conv2d/BiasAdd/ReadVariableOpЂ#EdgeNN/conv2d/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOpЂ&EdgeNN/conv2d_1/BiasAdd/ReadVariableOpЂ%EdgeNN/conv2d_1/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOpЂ-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOpЂ,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp
EdgeNN/summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
EdgeNN/summation/concat/axisЩ
EdgeNN/summation/concatConcatV2skel_imgnode_pos	node_pair%EdgeNN/summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
EdgeNN/summation/concat
&EdgeNN/summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2(
&EdgeNN/summation/Sum/reduction_indicesг
EdgeNN/summation/SumSum EdgeNN/summation/concat:output:0/EdgeNN/summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
EdgeNN/summation/Sumк
,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp
EdgeNN/conv2d_0_block1/Conv2DConv2DEdgeNN/summation/Sum:output:04EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block1/Conv2Dб
-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOpц
EdgeNN/conv2d_0_block1/BiasAddBiasAdd&EdgeNN/conv2d_0_block1/Conv2D:output:05EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2 
EdgeNN/conv2d_0_block1/BiasAddЊ
 EdgeNN/bn0_block1/ReadVariableOpReadVariableOp)edgenn_bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn0_block1/ReadVariableOpА
"EdgeNN/bn0_block1/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn0_block1/ReadVariableOp_1н
1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1й
"EdgeNN/bn0_block1/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block1/BiasAdd:output:0(EdgeNN/bn0_block1/ReadVariableOp:value:0*EdgeNN/bn0_block1/ReadVariableOp_1:value:09EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2$
"EdgeNN/bn0_block1/FusedBatchNormV3
EdgeNN/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
EdgeNN/concatenate/concat/axisу
EdgeNN/concatenate/concatConcatV2&EdgeNN/bn0_block1/FusedBatchNormV3:y:0	node_pair'EdgeNN/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
EdgeNN/concatenate/concat
EdgeNN/relu0_block1/ReluRelu"EdgeNN/concatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
EdgeNN/relu0_block1/Reluк
,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp
EdgeNN/conv2d_1_block1/Conv2DConv2D&EdgeNN/relu0_block1/Relu:activations:04EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block1/Conv2Dб
-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOpц
EdgeNN/conv2d_1_block1/BiasAddBiasAdd&EdgeNN/conv2d_1_block1/Conv2D:output:05EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2 
EdgeNN/conv2d_1_block1/BiasAddЊ
 EdgeNN/bn1_block1/ReadVariableOpReadVariableOp)edgenn_bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn1_block1/ReadVariableOpА
"EdgeNN/bn1_block1/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn1_block1/ReadVariableOp_1н
1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1й
"EdgeNN/bn1_block1/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block1/BiasAdd:output:0(EdgeNN/bn1_block1/ReadVariableOp:value:0*EdgeNN/bn1_block1/ReadVariableOp_1:value:09EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2$
"EdgeNN/bn1_block1/FusedBatchNormV3
 EdgeNN/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2"
 EdgeNN/concatenate_1/concat/axisщ
EdgeNN/concatenate_1/concatConcatV2&EdgeNN/bn1_block1/FusedBatchNormV3:y:0	node_pair)EdgeNN/concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
EdgeNN/concatenate_1/concat
EdgeNN/relu1_block1/ReluRelu$EdgeNN/concatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
EdgeNN/relu1_block1/Reluо
EdgeNN/max_pooling2d/MaxPoolMaxPool&EdgeNN/relu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
EdgeNN/max_pooling2d/MaxPoolк
,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp
EdgeNN/conv2d_0_block2/Conv2DConv2D%EdgeNN/max_pooling2d/MaxPool:output:04EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block2/Conv2Dб
-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOpц
EdgeNN/conv2d_0_block2/BiasAddBiasAdd&EdgeNN/conv2d_0_block2/Conv2D:output:05EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2 
EdgeNN/conv2d_0_block2/BiasAddЊ
 EdgeNN/bn0_block2/ReadVariableOpReadVariableOp)edgenn_bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn0_block2/ReadVariableOpА
"EdgeNN/bn0_block2/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn0_block2/ReadVariableOp_1н
1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1й
"EdgeNN/bn0_block2/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block2/BiasAdd:output:0(EdgeNN/bn0_block2/ReadVariableOp:value:0*EdgeNN/bn0_block2/ReadVariableOp_1:value:09EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2$
"EdgeNN/bn0_block2/FusedBatchNormV3 
EdgeNN/relu0_block2/ReluRelu&EdgeNN/bn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
EdgeNN/relu0_block2/Reluк
,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp
EdgeNN/conv2d_1_block2/Conv2DConv2D&EdgeNN/relu0_block2/Relu:activations:04EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block2/Conv2Dб
-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOpц
EdgeNN/conv2d_1_block2/BiasAddBiasAdd&EdgeNN/conv2d_1_block2/Conv2D:output:05EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2 
EdgeNN/conv2d_1_block2/BiasAddЊ
 EdgeNN/bn1_block2/ReadVariableOpReadVariableOp)edgenn_bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn1_block2/ReadVariableOpА
"EdgeNN/bn1_block2/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn1_block2/ReadVariableOp_1н
1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1й
"EdgeNN/bn1_block2/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block2/BiasAdd:output:0(EdgeNN/bn1_block2/ReadVariableOp:value:0*EdgeNN/bn1_block2/ReadVariableOp_1:value:09EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2$
"EdgeNN/bn1_block2/FusedBatchNormV3 
EdgeNN/relu1_block2/ReluRelu&EdgeNN/bn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
EdgeNN/relu1_block2/Reluр
EdgeNN/max_pooling2d_1/MaxPoolMaxPool&EdgeNN/relu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2 
EdgeNN/max_pooling2d_1/MaxPoolк
,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp
EdgeNN/conv2d_0_block3/Conv2DConv2D'EdgeNN/max_pooling2d_1/MaxPool:output:04EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block3/Conv2Dб
-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOpф
EdgeNN/conv2d_0_block3/BiasAddBiasAdd&EdgeNN/conv2d_0_block3/Conv2D:output:05EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2 
EdgeNN/conv2d_0_block3/BiasAddЊ
 EdgeNN/bn0_block3/ReadVariableOpReadVariableOp)edgenn_bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn0_block3/ReadVariableOpА
"EdgeNN/bn0_block3/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn0_block3/ReadVariableOp_1н
1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1з
"EdgeNN/bn0_block3/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block3/BiasAdd:output:0(EdgeNN/bn0_block3/ReadVariableOp:value:0*EdgeNN/bn0_block3/ReadVariableOp_1:value:09EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2$
"EdgeNN/bn0_block3/FusedBatchNormV3
EdgeNN/relu0_block3/ReluRelu&EdgeNN/bn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
EdgeNN/relu0_block3/Reluк
,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp
EdgeNN/conv2d_1_block3/Conv2DConv2D&EdgeNN/relu0_block3/Relu:activations:04EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block3/Conv2Dб
-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOpф
EdgeNN/conv2d_1_block3/BiasAddBiasAdd&EdgeNN/conv2d_1_block3/Conv2D:output:05EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2 
EdgeNN/conv2d_1_block3/BiasAddЊ
 EdgeNN/bn1_block3/ReadVariableOpReadVariableOp)edgenn_bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02"
 EdgeNN/bn1_block3/ReadVariableOpА
"EdgeNN/bn1_block3/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"EdgeNN/bn1_block3/ReadVariableOp_1н
1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1з
"EdgeNN/bn1_block3/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block3/BiasAdd:output:0(EdgeNN/bn1_block3/ReadVariableOp:value:0*EdgeNN/bn1_block3/ReadVariableOp_1:value:09EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2$
"EdgeNN/bn1_block3/FusedBatchNormV3
EdgeNN/relu1_block3/ReluRelu&EdgeNN/bn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
EdgeNN/relu1_block3/ReluП
#EdgeNN/conv2d/Conv2D/ReadVariableOpReadVariableOp,edgenn_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#EdgeNN/conv2d/Conv2D/ReadVariableOpэ
EdgeNN/conv2d/Conv2DConv2D&EdgeNN/relu1_block3/Relu:activations:0+EdgeNN/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
EdgeNN/conv2d/Conv2DЖ
$EdgeNN/conv2d/BiasAdd/ReadVariableOpReadVariableOp-edgenn_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$EdgeNN/conv2d/BiasAdd/ReadVariableOpР
EdgeNN/conv2d/BiasAddBiasAddEdgeNN/conv2d/Conv2D:output:0,EdgeNN/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
EdgeNN/conv2d/BiasAddХ
)EdgeNN/batch_normalization/ReadVariableOpReadVariableOp2edgenn_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02+
)EdgeNN/batch_normalization/ReadVariableOpЫ
+EdgeNN/batch_normalization/ReadVariableOp_1ReadVariableOp4edgenn_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02-
+EdgeNN/batch_normalization/ReadVariableOp_1ј
:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpCedgenn_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02<
:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOpў
<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEedgenn_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02>
<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1
+EdgeNN/batch_normalization/FusedBatchNormV3FusedBatchNormV3EdgeNN/conv2d/BiasAdd:output:01EdgeNN/batch_normalization/ReadVariableOp:value:03EdgeNN/batch_normalization/ReadVariableOp_1:value:0BEdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0DEdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2-
+EdgeNN/batch_normalization/FusedBatchNormV3Ћ
EdgeNN/relu_C3_block3/ReluRelu/EdgeNN/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
EdgeNN/relu_C3_block3/Reluт
EdgeNN/max_pooling2d_2/MaxPoolMaxPool(EdgeNN/relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2 
EdgeNN/max_pooling2d_2/MaxPoolк
,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02.
,EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp
EdgeNN/conv2d_0_block4/Conv2DConv2D'EdgeNN/max_pooling2d_2/MaxPool:output:04EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
EdgeNN/conv2d_0_block4/Conv2Dб
-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOpф
EdgeNN/conv2d_0_block4/BiasAddBiasAdd&EdgeNN/conv2d_0_block4/Conv2D:output:05EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02 
EdgeNN/conv2d_0_block4/BiasAddЊ
 EdgeNN/bn0_block4/ReadVariableOpReadVariableOp)edgenn_bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02"
 EdgeNN/bn0_block4/ReadVariableOpА
"EdgeNN/bn0_block4/ReadVariableOp_1ReadVariableOp+edgenn_bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02$
"EdgeNN/bn0_block4/ReadVariableOp_1н
1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype023
1EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1з
"EdgeNN/bn0_block4/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_0_block4/BiasAdd:output:0(EdgeNN/bn0_block4/ReadVariableOp:value:0*EdgeNN/bn0_block4/ReadVariableOp_1:value:09EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2$
"EdgeNN/bn0_block4/FusedBatchNormV3
EdgeNN/relu0_block4/ReluRelu&EdgeNN/bn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
EdgeNN/relu0_block4/Reluк
,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp5edgenn_conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02.
,EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp
EdgeNN/conv2d_1_block4/Conv2DConv2D&EdgeNN/relu0_block4/Relu:activations:04EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
EdgeNN/conv2d_1_block4/Conv2Dб
-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp6edgenn_conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02/
-EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOpф
EdgeNN/conv2d_1_block4/BiasAddBiasAdd&EdgeNN/conv2d_1_block4/Conv2D:output:05EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02 
EdgeNN/conv2d_1_block4/BiasAddЊ
 EdgeNN/bn1_block4/ReadVariableOpReadVariableOp)edgenn_bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02"
 EdgeNN/bn1_block4/ReadVariableOpА
"EdgeNN/bn1_block4/ReadVariableOp_1ReadVariableOp+edgenn_bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02$
"EdgeNN/bn1_block4/ReadVariableOp_1н
1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp:edgenn_bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype023
1EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOpу
3EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<edgenn_bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype025
3EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1з
"EdgeNN/bn1_block4/FusedBatchNormV3FusedBatchNormV3'EdgeNN/conv2d_1_block4/BiasAdd:output:0(EdgeNN/bn1_block4/ReadVariableOp:value:0*EdgeNN/bn1_block4/ReadVariableOp_1:value:09EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp:value:0;EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2$
"EdgeNN/bn1_block4/FusedBatchNormV3
EdgeNN/relu1_block4/ReluRelu&EdgeNN/bn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
EdgeNN/relu1_block4/ReluХ
%EdgeNN/conv2d_1/Conv2D/ReadVariableOpReadVariableOp.edgenn_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%EdgeNN/conv2d_1/Conv2D/ReadVariableOpѓ
EdgeNN/conv2d_1/Conv2DConv2D&EdgeNN/relu1_block4/Relu:activations:0-EdgeNN/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
EdgeNN/conv2d_1/Conv2DМ
&EdgeNN/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp/edgenn_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&EdgeNN/conv2d_1/BiasAdd/ReadVariableOpШ
EdgeNN/conv2d_1/BiasAddBiasAddEdgeNN/conv2d_1/Conv2D:output:0.EdgeNN/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
EdgeNN/conv2d_1/BiasAddЫ
+EdgeNN/batch_normalization_1/ReadVariableOpReadVariableOp4edgenn_batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02-
+EdgeNN/batch_normalization_1/ReadVariableOpб
-EdgeNN/batch_normalization_1/ReadVariableOp_1ReadVariableOp6edgenn_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02/
-EdgeNN/batch_normalization_1/ReadVariableOp_1ў
<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpEedgenn_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02>
<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGedgenn_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02@
>EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1
-EdgeNN/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 EdgeNN/conv2d_1/BiasAdd:output:03EdgeNN/batch_normalization_1/ReadVariableOp:value:05EdgeNN/batch_normalization_1/ReadVariableOp_1:value:0DEdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0FEdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2/
-EdgeNN/batch_normalization_1/FusedBatchNormV3­
EdgeNN/relu_C3_block4/ReluRelu1EdgeNN/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
EdgeNN/relu_C3_block4/Reluт
EdgeNN/max_pooling2d_3/MaxPoolMaxPool(EdgeNN/relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2 
EdgeNN/max_pooling2d_3/MaxPoolЗ
1EdgeNN/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1EdgeNN/global_max_pooling2d/Max/reduction_indicesљ
EdgeNN/global_max_pooling2d/MaxMax'EdgeNN/max_pooling2d_3/MaxPool:output:0:EdgeNN/global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2!
EdgeNN/global_max_pooling2d/Max
#EdgeNN/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2%
#EdgeNN/conv1d/conv1d/ExpandDims/dimц
EdgeNN/conv1d/conv1d/ExpandDims
ExpandDims(EdgeNN/global_max_pooling2d/Max:output:0,EdgeNN/conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02!
EdgeNN/conv1d/conv1d/ExpandDimsт
0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9edgenn_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype022
0EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp
%EdgeNN/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%EdgeNN/conv1d/conv1d/ExpandDims_1/dimя
!EdgeNN/conv1d/conv1d/ExpandDims_1
ExpandDims8EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0.EdgeNN/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:02#
!EdgeNN/conv1d/conv1d/ExpandDims_1
EdgeNN/conv1d/conv1d/ShapeShape(EdgeNN/conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
EdgeNN/conv1d/conv1d/Shape
(EdgeNN/conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(EdgeNN/conv1d/conv1d/strided_slice/stackЋ
*EdgeNN/conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2,
*EdgeNN/conv1d/conv1d/strided_slice/stack_1Ђ
*EdgeNN/conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*EdgeNN/conv1d/conv1d/strided_slice/stack_2о
"EdgeNN/conv1d/conv1d/strided_sliceStridedSlice#EdgeNN/conv1d/conv1d/Shape:output:01EdgeNN/conv1d/conv1d/strided_slice/stack:output:03EdgeNN/conv1d/conv1d/strided_slice/stack_1:output:03EdgeNN/conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"EdgeNN/conv1d/conv1d/strided_sliceЁ
"EdgeNN/conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2$
"EdgeNN/conv1d/conv1d/Reshape/shapeи
EdgeNN/conv1d/conv1d/ReshapeReshape(EdgeNN/conv1d/conv1d/ExpandDims:output:0+EdgeNN/conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
EdgeNN/conv1d/conv1d/Reshapeњ
EdgeNN/conv1d/conv1d/Conv2DConv2D%EdgeNN/conv1d/conv1d/Reshape:output:0*EdgeNN/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
EdgeNN/conv1d/conv1d/Conv2DЁ
$EdgeNN/conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2&
$EdgeNN/conv1d/conv1d/concat/values_1
 EdgeNN/conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 EdgeNN/conv1d/conv1d/concat/axisћ
EdgeNN/conv1d/conv1d/concatConcatV2+EdgeNN/conv1d/conv1d/strided_slice:output:0-EdgeNN/conv1d/conv1d/concat/values_1:output:0)EdgeNN/conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
EdgeNN/conv1d/conv1d/concatе
EdgeNN/conv1d/conv1d/Reshape_1Reshape$EdgeNN/conv1d/conv1d/Conv2D:output:0$EdgeNN/conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2 
EdgeNN/conv1d/conv1d/Reshape_1Ъ
EdgeNN/conv1d/conv1d/SqueezeSqueeze'EdgeNN/conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
EdgeNN/conv1d/conv1d/SqueezeЅ
&EdgeNN/conv1d/squeeze_batch_dims/ShapeShape%EdgeNN/conv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2(
&EdgeNN/conv1d/squeeze_batch_dims/ShapeЖ
4EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stackУ
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ28
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_1К
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_2І
.EdgeNN/conv1d/squeeze_batch_dims/strided_sliceStridedSlice/EdgeNN/conv1d/squeeze_batch_dims/Shape:output:0=EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack:output:0?EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_1:output:0?EdgeNN/conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask20
.EdgeNN/conv1d/squeeze_batch_dims/strided_sliceЕ
.EdgeNN/conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      20
.EdgeNN/conv1d/squeeze_batch_dims/Reshape/shapeѕ
(EdgeNN/conv1d/squeeze_batch_dims/ReshapeReshape%EdgeNN/conv1d/conv1d/Squeeze:output:07EdgeNN/conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2*
(EdgeNN/conv1d/squeeze_batch_dims/Reshapeя
7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp@edgenn_conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp
(EdgeNN/conv1d/squeeze_batch_dims/BiasAddBiasAdd1EdgeNN/conv1d/squeeze_batch_dims/Reshape:output:0?EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2*
(EdgeNN/conv1d/squeeze_batch_dims/BiasAddЕ
0EdgeNN/conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      22
0EdgeNN/conv1d/squeeze_batch_dims/concat/values_1Ї
,EdgeNN/conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,EdgeNN/conv1d/squeeze_batch_dims/concat/axisЗ
'EdgeNN/conv1d/squeeze_batch_dims/concatConcatV27EdgeNN/conv1d/squeeze_batch_dims/strided_slice:output:09EdgeNN/conv1d/squeeze_batch_dims/concat/values_1:output:05EdgeNN/conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'EdgeNN/conv1d/squeeze_batch_dims/concat
*EdgeNN/conv1d/squeeze_batch_dims/Reshape_1Reshape1EdgeNN/conv1d/squeeze_batch_dims/BiasAdd:output:00EdgeNN/conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2,
*EdgeNN/conv1d/squeeze_batch_dims/Reshape_1Ј
EdgeNN/conv1d/SigmoidSigmoid3EdgeNN/conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
EdgeNN/conv1d/SigmoidР
&EdgeNN/tf.compat.v1.squeeze/adj_outputSqueezeEdgeNN/conv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2(
&EdgeNN/tf.compat.v1.squeeze/adj_output
IdentityIdentity/EdgeNN/tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp;^EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp=^EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*^EdgeNN/batch_normalization/ReadVariableOp,^EdgeNN/batch_normalization/ReadVariableOp_1=^EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?^EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1,^EdgeNN/batch_normalization_1/ReadVariableOp.^EdgeNN/batch_normalization_1/ReadVariableOp_12^EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block1/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block1/ReadVariableOp#^EdgeNN/bn0_block1/ReadVariableOp_12^EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block2/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block2/ReadVariableOp#^EdgeNN/bn0_block2/ReadVariableOp_12^EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block3/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block3/ReadVariableOp#^EdgeNN/bn0_block3/ReadVariableOp_12^EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn0_block4/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn0_block4/ReadVariableOp#^EdgeNN/bn0_block4/ReadVariableOp_12^EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block1/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block1/ReadVariableOp#^EdgeNN/bn1_block1/ReadVariableOp_12^EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block2/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block2/ReadVariableOp#^EdgeNN/bn1_block2/ReadVariableOp_12^EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block3/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block3/ReadVariableOp#^EdgeNN/bn1_block3/ReadVariableOp_12^EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp4^EdgeNN/bn1_block4/FusedBatchNormV3/ReadVariableOp_1!^EdgeNN/bn1_block4/ReadVariableOp#^EdgeNN/bn1_block4/ReadVariableOp_11^EdgeNN/conv1d/conv1d/ExpandDims_1/ReadVariableOp8^EdgeNN/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp%^EdgeNN/conv2d/BiasAdd/ReadVariableOp$^EdgeNN/conv2d/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block1/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block1/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block2/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block2/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block3/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block3/Conv2D/ReadVariableOp.^EdgeNN/conv2d_0_block4/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_0_block4/Conv2D/ReadVariableOp'^EdgeNN/conv2d_1/BiasAdd/ReadVariableOp&^EdgeNN/conv2d_1/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block1/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block1/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block2/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block2/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block3/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block3/Conv2D/ReadVariableOp.^EdgeNN/conv2d_1_block4/BiasAdd/ReadVariableOp-^EdgeNN/conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp:EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp2|
<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<EdgeNN/batch_normalization/FusedBatchNormV3/ReadVariableOp_12V
)EdgeNN/batch_normalization/ReadVariableOp)EdgeNN/batch_normalization/ReadVariableOp2Z
+EdgeNN/batch_normalization/ReadVariableOp_1+EdgeNN/batch_normalization/ReadVariableOp_12|
<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp<EdgeNN/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
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
:џџџџџџџџџ
"
_user_specified_name
skel_img:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
node_pos:\X
1
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	node_pair
у
G
+__inference_relu0_block3_layer_call_fn_8096

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ђ
є
)__inference_bn1_block4_layer_call_fn_8784

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs

Г
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7978

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж

D__inference_bn0_block1_layer_call_and_return_conditional_losses_7196

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Љ
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7907

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
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
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
њ
є
)__inference_bn0_block2_layer_call_fn_7700

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
J
.__inference_max_pooling2d_3_layer_call_fn_9006

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
ж

D__inference_bn1_block3_layer_call_and_return_conditional_losses_8134

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Г
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8520

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs


I__inference_conv2d_1_block3_layer_call_and_return_conditional_losses_8106

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ў

)__inference_bn0_block2_layer_call_fn_7718

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
у
G
+__inference_relu1_block3_layer_call_fn_8270

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ц
џ
4__inference_batch_normalization_1_layer_call_fn_8922

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
п
Ь7
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
identityЂ"batch_normalization/AssignNewValueЂ$batch_normalization/AssignNewValue_1Ђ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ$batch_normalization_1/AssignNewValueЂ&batch_normalization_1/AssignNewValue_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђbn0_block1/AssignNewValueЂbn0_block1/AssignNewValue_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђbn0_block2/AssignNewValueЂbn0_block2/AssignNewValue_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђbn0_block3/AssignNewValueЂbn0_block3/AssignNewValue_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђbn0_block4/AssignNewValueЂbn0_block4/AssignNewValue_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђbn1_block1/AssignNewValueЂbn1_block1/AssignNewValue_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђbn1_block2/AssignNewValueЂbn1_block2/AssignNewValue_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђbn1_block3/AssignNewValueЂbn1_block3/AssignNewValue_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђbn1_block4/AssignNewValueЂbn1_block4/AssignNewValue_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisГ
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block1/FusedBatchNormV3љ
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue
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
concatenate/concat/axisЦ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block1/FusedBatchNormV3љ
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue
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
concatenate_1/concat/axisЬ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block2/FusedBatchNormV3љ
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block2/FusedBatchNormV3љ
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block3/FusedBatchNormV3љ
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block3/FusedBatchNormV3љ
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1с
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2&
$batch_normalization/FusedBatchNormV3І
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueВ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block4/FusedBatchNormV3љ
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block4/FusedBatchNormV3љ
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1я
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2(
&batch_normalization_1/FusedBatchNormV3А
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueМ
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЯ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
ж

D__inference_bn0_block2_layer_call_and_return_conditional_losses_7592

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п1
Е
@__inference_conv1d_layer_call_and_return_conditional_losses_9068

inputsA
+conv1d_expanddims_1_readvariableop_resource:0@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identityЂ"conv1d/ExpandDims_1/ReadVariableOpЂ)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
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
conv1d/Shape
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2
conv1d/strided_slice/stack_1
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/Reshape/shape 
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/ReshapeТ
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/Conv2D
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
џџџџџџџџџ2
conv1d/concat/axisЕ
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/Reshape_1 
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stackЇ
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ2*
(squeeze_batch_dims/strided_slice/stack_1
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2в
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2"
 squeeze_batch_dims/Reshape/shapeН
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
squeeze_batch_dims/ReshapeХ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOpб
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
squeeze_batch_dims/BiasAdd
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
squeeze_batch_dims/concat/axisё
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concatЪ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
squeeze_batch_dims/Reshape_1~
SigmoidSigmoid%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ0: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
ђ
є
)__inference_bn0_block3_layer_call_fn_8068

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
я

)__inference_bn0_block1_layer_call_fn_7286

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

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
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1


D__inference_bn0_block3_layer_call_and_return_conditional_losses_7996

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs


I__inference_conv2d_1_block1_layer_call_and_return_conditional_losses_7356

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


D__inference_bn0_block2_layer_call_and_return_conditional_losses_7628

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
Г
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8556

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
А
j
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9018

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices
MaxMaxinputsMax/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
Maxh
IdentityIdentityMax:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ0:W S
/
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
у
G
+__inference_relu1_block4_layer_call_fn_8812

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
ж

D__inference_bn1_block2_layer_call_and_return_conditional_losses_7766

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ў
b
F__inference_relu1_block3_layer_call_and_return_conditional_losses_8265

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Щ
Г
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7646

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я

)__inference_bn1_block4_layer_call_fn_8766

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
ў
b
F__inference_relu0_block3_layer_call_and_return_conditional_losses_8091

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

Г
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7402

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Г
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7784

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
я

)__inference_bn0_block3_layer_call_fn_8050

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ѕ
J
.__inference_max_pooling2d_1_layer_call_fn_7922

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
Й
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
identityЂStatefulPartitionedCallЊ	
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
:џџџџџџџџџ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_11942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	node_pair:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
node_pos:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
skel_img
С
Г
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8730

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
њ

ч
.__inference_conv2d_0_block2_layer_call_fn_7574

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
є
)__inference_bn1_block3_layer_call_fn_8206

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Л
є
)__inference_bn0_block3_layer_call_fn_8032

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ъ
М
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8362

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ђ
є
)__inference_bn0_block4_layer_call_fn_8610

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
я

)__inference_bn1_block3_layer_call_fn_8224

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ф
§
2__inference_batch_normalization_layer_call_fn_8380

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ

ч
.__inference_conv2d_1_block2_layer_call_fn_7748

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 
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
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
Ў

)__inference_bn1_block1_layer_call_fn_7510

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
G
+__inference_relu0_block1_layer_call_fn_7346

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

b
F__inference_relu1_block2_layer_call_and_return_conditional_losses_7897

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


I__inference_conv2d_1_block4_layer_call_and_return_conditional_losses_8648

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
Ф1

%__inference_conv1d_layer_call_fn_9106

inputsA
+conv1d_expanddims_1_readvariableop_resource:0@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identityЂ"conv1d/ExpandDims_1/ReadVariableOpЂ)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
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
conv1d/Shape
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2
conv1d/strided_slice/stack_1
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/Reshape/shape 
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/ReshapeТ
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/Conv2D
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
џџџџџџџџџ2
conv1d/concat/axisЕ
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/Reshape_1 
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stackЇ
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ2*
(squeeze_batch_dims/strided_slice/stack_1
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2в
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2"
 squeeze_batch_dims/Reshape/shapeН
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
squeeze_batch_dims/ReshapeХ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOpб
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
squeeze_batch_dims/BiasAdd
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
squeeze_batch_dims/concat/axisё
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concatЪ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
squeeze_batch_dims/Reshape_1~
SigmoidSigmoid%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ0: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
У	
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
џџџџџџџџџ2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Sum/reduction_indices
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2

Г
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8694

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs

H
,__inference_max_pooling2d_layer_call_fn_7549

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

b
F__inference_relu1_block1_layer_call_and_return_conditional_losses_7529

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


I__inference_conv2d_1_block2_layer_call_and_return_conditional_losses_7738

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
є
)__inference_bn1_block3_layer_call_fn_8242

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ј	
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
џџџџџџџџџ2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Sum/reduction_indices
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Л
є
)__inference_bn0_block1_layer_call_fn_7268

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


M__inference_batch_normalization_layer_call_and_return_conditional_losses_8344

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
я

)__inference_bn0_block4_layer_call_fn_8592

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
ћ
§
2__inference_batch_normalization_layer_call_fn_8416

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
њ
є
)__inference_bn1_block1_layer_call_fn_7492

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


D__inference_bn1_block3_layer_call_and_return_conditional_losses_8170

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs


I__inference_conv2d_0_block4_layer_call_and_return_conditional_losses_8474

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs


D__inference_bn1_block1_layer_call_and_return_conditional_losses_7420

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч

о
%__inference_conv2d_layer_call_fn_8290

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs


D__inference_bn1_block4_layer_call_and_return_conditional_losses_8712

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
х
I
-__inference_relu_C3_block4_layer_call_fn_8986

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
І

)__inference_bn0_block3_layer_call_fn_8086

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

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
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
њ

ч
.__inference_conv2d_0_block1_layer_call_fn_7178

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
G
+__inference_relu1_block1_layer_call_fn_7534

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
G
+__inference_relu1_block2_layer_call_fn_7902

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
Г
D__inference_bn0_block3_layer_call_and_return_conditional_losses_8014

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Љ
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8449

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Џ
Ё
2__inference_batch_normalization_layer_call_fn_8434

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Го
р2
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
identityЂ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisГ
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЦ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisЬ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block2/FusedBatchNormV3
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block2/FusedBatchNormV3
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1І
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn0_block3/FusedBatchNormV3
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1І
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn1_block3/FusedBatchNormV3
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1г
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1І
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn0_block4/FusedBatchNormV3
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1І
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn1_block4/FusedBatchNormV3
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityу
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Ь
О
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8904

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs

Г
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7214

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І

)__inference_bn1_block3_layer_call_fn_8260

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ж

D__inference_bn0_block4_layer_call_and_return_conditional_losses_8502

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs


D__inference_bn0_block4_layer_call_and_return_conditional_losses_8538

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
ж

D__inference_bn0_block3_layer_call_and_return_conditional_losses_7960

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ў
b
F__inference_relu0_block4_layer_call_and_return_conditional_losses_8633

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
Л
є
)__inference_bn0_block4_layer_call_fn_8574

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
І

)__inference_bn1_block4_layer_call_fn_8802

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs

М
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8326

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
я

)__inference_bn1_block1_layer_call_fn_7474

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Юо
ћ2
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
identityЂ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisГ
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЦ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisЬ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block2/FusedBatchNormV3
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block2/FusedBatchNormV3
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1І
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn0_block3/FusedBatchNormV3
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1І
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn1_block3/FusedBatchNormV3
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1г
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1І
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn0_block4/FusedBatchNormV3
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1І
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn1_block4/FusedBatchNormV3
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityу
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Щ
Г
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7250

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
У	
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
џџџџџџџџџ2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Sum/reduction_indices
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sumj
IdentityIdentitySum:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:[ W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2

O
3__inference_global_max_pooling2d_layer_call_fn_9030

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices
MaxMaxinputsMax/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
Maxh
IdentityIdentityMax:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ0:W S
/
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs

О
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8868

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
ы
G
+__inference_relu0_block2_layer_call_fn_7728

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я

)__inference_bn0_block2_layer_call_fn_7682

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8886

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
ф
Э7
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
identityЂ"batch_normalization/AssignNewValueЂ$batch_normalization/AssignNewValue_1Ђ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ$batch_normalization_1/AssignNewValueЂ&batch_normalization_1/AssignNewValue_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђbn0_block1/AssignNewValueЂbn0_block1/AssignNewValue_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђbn0_block2/AssignNewValueЂbn0_block2/AssignNewValue_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђbn0_block3/AssignNewValueЂbn0_block3/AssignNewValue_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђbn0_block4/AssignNewValueЂbn0_block4/AssignNewValue_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђbn1_block1/AssignNewValueЂbn1_block1/AssignNewValue_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђbn1_block2/AssignNewValueЂbn1_block2/AssignNewValue_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђbn1_block3/AssignNewValueЂbn1_block3/AssignNewValue_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђbn1_block4/AssignNewValueЂbn1_block4/AssignNewValue_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisД
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block1/FusedBatchNormV3љ
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue
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
concatenate/concat/axisЧ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block1/FusedBatchNormV3љ
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue
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
concatenate_1/concat/axisЭ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block2/FusedBatchNormV3љ
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block2/FusedBatchNormV3љ
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block3/FusedBatchNormV3љ
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block3/FusedBatchNormV3љ
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1с
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2&
$batch_normalization/FusedBatchNormV3І
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueВ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block4/FusedBatchNormV3љ
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block4/FusedBatchNormV3љ
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1я
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2(
&batch_normalization_1/FusedBatchNormV3А
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueМ
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЯ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:џџџџџџџџџ
"
_user_specified_name
skel_img:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
node_pos:\X
1
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	node_pair
ј
Ё
2__inference_batch_normalization_layer_call_fn_8398

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І

)__inference_bn0_block4_layer_call_fn_8628

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs

J
.__inference_max_pooling2d_1_layer_call_fn_7917

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
є
)__inference_bn0_block1_layer_call_fn_7304

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х
I
-__inference_relu_C3_block3_layer_call_fn_8444

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Б
Ѓ
4__inference_batch_normalization_1_layer_call_fn_8976

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
го
ќ2
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
identityЂ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisД
summation/concatConcatV2skel_imgnode_pos	node_pairsummation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block1/FusedBatchNormV3t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЧ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0	node_pair concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block1/FusedBatchNormV3x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisЭ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0	node_pair"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn0_block2/FusedBatchNormV3
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ј
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
bn1_block2/FusedBatchNormV3
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1І
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn0_block3/FusedBatchNormV3
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1І
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2
bn1_block3/FusedBatchNormV3
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1г
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1І
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn0_block4/FusedBatchNormV3
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1І
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
bn1_block4/FusedBatchNormV3
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1с
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityу
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
:џџџџџџџџџ
"
_user_specified_name
skel_img:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
node_pos:\X
1
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	node_pair
Ї
H
,__inference_max_pooling2d_layer_call_fn_7554

inputs
identity
MaxPoolMaxPoolinputs*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8996

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs


D__inference_bn1_block2_layer_call_and_return_conditional_losses_7802

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ћ
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8822

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ  0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
ј
j
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9012

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indices
MaxMaxinputsMax/reduction_indices:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
	keep_dims(2
Maxq
IdentityIdentityMax:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


I__inference_conv2d_0_block2_layer_call_and_return_conditional_losses_7564

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7544

inputs
identity
MaxPoolMaxPoolinputs*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
є
)__inference_bn1_block2_layer_call_fn_7838

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п

M__inference_batch_normalization_layer_call_and_return_conditional_losses_8308

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

d
H__inference_relu_C3_block3_layer_call_and_return_conditional_losses_8439

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

J
.__inference_max_pooling2d_3_layer_call_fn_9001

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7912

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

b
F__inference_relu0_block1_layer_call_and_return_conditional_losses_7341

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:џџџџџџџџџ2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў
b
F__inference_relu1_block4_layer_call_and_return_conditional_losses_8807

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ  02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  0:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
њ
є
)__inference_bn1_block2_layer_call_fn_7874

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
џ
4__inference_batch_normalization_1_layer_call_fn_8958

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ  02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ  0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ  0
 
_user_specified_nameinputs
я

)__inference_bn1_block2_layer_call_fn_7856

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identityм
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


D__inference_bn0_block1_layer_call_and_return_conditional_losses_7232

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:џџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
J
.__inference_max_pooling2d_2_layer_call_fn_8464

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
њ

ч
.__inference_conv2d_1_block1_layer_call_fn_7366

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж

D__inference_bn1_block4_layer_call_and_return_conditional_losses_8676

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
њ
ч7
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
identityЂ"batch_normalization/AssignNewValueЂ$batch_normalization/AssignNewValue_1Ђ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ$batch_normalization_1/AssignNewValueЂ&batch_normalization_1/AssignNewValue_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђbn0_block1/AssignNewValueЂbn0_block1/AssignNewValue_1Ђ*bn0_block1/FusedBatchNormV3/ReadVariableOpЂ,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block1/ReadVariableOpЂbn0_block1/ReadVariableOp_1Ђbn0_block2/AssignNewValueЂbn0_block2/AssignNewValue_1Ђ*bn0_block2/FusedBatchNormV3/ReadVariableOpЂ,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block2/ReadVariableOpЂbn0_block2/ReadVariableOp_1Ђbn0_block3/AssignNewValueЂbn0_block3/AssignNewValue_1Ђ*bn0_block3/FusedBatchNormV3/ReadVariableOpЂ,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block3/ReadVariableOpЂbn0_block3/ReadVariableOp_1Ђbn0_block4/AssignNewValueЂbn0_block4/AssignNewValue_1Ђ*bn0_block4/FusedBatchNormV3/ReadVariableOpЂ,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn0_block4/ReadVariableOpЂbn0_block4/ReadVariableOp_1Ђbn1_block1/AssignNewValueЂbn1_block1/AssignNewValue_1Ђ*bn1_block1/FusedBatchNormV3/ReadVariableOpЂ,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block1/ReadVariableOpЂbn1_block1/ReadVariableOp_1Ђbn1_block2/AssignNewValueЂbn1_block2/AssignNewValue_1Ђ*bn1_block2/FusedBatchNormV3/ReadVariableOpЂ,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block2/ReadVariableOpЂbn1_block2/ReadVariableOp_1Ђbn1_block3/AssignNewValueЂbn1_block3/AssignNewValue_1Ђ*bn1_block3/FusedBatchNormV3/ReadVariableOpЂ,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block3/ReadVariableOpЂbn1_block3/ReadVariableOp_1Ђbn1_block4/AssignNewValueЂbn1_block4/AssignNewValue_1Ђ*bn1_block4/FusedBatchNormV3/ReadVariableOpЂ,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Ђbn1_block4/ReadVariableOpЂbn1_block4/ReadVariableOp_1Ђ)conv1d/conv1d/ExpandDims_1/ReadVariableOpЂ0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ&conv2d_0_block1/BiasAdd/ReadVariableOpЂ%conv2d_0_block1/Conv2D/ReadVariableOpЂ&conv2d_0_block2/BiasAdd/ReadVariableOpЂ%conv2d_0_block2/Conv2D/ReadVariableOpЂ&conv2d_0_block3/BiasAdd/ReadVariableOpЂ%conv2d_0_block3/Conv2D/ReadVariableOpЂ&conv2d_0_block4/BiasAdd/ReadVariableOpЂ%conv2d_0_block4/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ&conv2d_1_block1/BiasAdd/ReadVariableOpЂ%conv2d_1_block1/Conv2D/ReadVariableOpЂ&conv2d_1_block2/BiasAdd/ReadVariableOpЂ%conv2d_1_block2/Conv2D/ReadVariableOpЂ&conv2d_1_block3/BiasAdd/ReadVariableOpЂ%conv2d_1_block3/Conv2D/ReadVariableOpЂ&conv2d_1_block4/BiasAdd/ReadVariableOpЂ%conv2d_1_block4/Conv2D/ReadVariableOpy
summation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
summation/concat/axisГ
summation/concatConcatV2inputs_0inputs_1inputs_2summation/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
summation/concat
summation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
summation/Sum/reduction_indicesЗ
summation/SumSumsummation/concat:output:0(summation/Sum/reduction_indices:output:0*
T0*1
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
summation/SumХ
%conv2d_0_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block1/Conv2D/ReadVariableOpх
conv2d_0_block1/Conv2DConv2Dsummation/Sum:output:0-conv2d_0_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block1/Conv2DМ
&conv2d_0_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block1/BiasAdd/ReadVariableOpЪ
conv2d_0_block1/BiasAddBiasAddconv2d_0_block1/Conv2D:output:0.conv2d_0_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block1/BiasAdd
bn0_block1/ReadVariableOpReadVariableOp"bn0_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp
bn0_block1/ReadVariableOp_1ReadVariableOp$bn0_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block1/ReadVariableOp_1Ш
*bn0_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block1/BiasAdd:output:0!bn0_block1/ReadVariableOp:value:0#bn0_block1/ReadVariableOp_1:value:02bn0_block1/FusedBatchNormV3/ReadVariableOp:value:04bn0_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block1/FusedBatchNormV3љ
bn0_block1/AssignNewValueAssignVariableOp3bn0_block1_fusedbatchnormv3_readvariableop_resource(bn0_block1/FusedBatchNormV3:batch_mean:0+^bn0_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block1/AssignNewValue
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
concatenate/concat/axisЦ
concatenate/concatConcatV2bn0_block1/FusedBatchNormV3:y:0inputs_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate/concat
relu0_block1/ReluReluconcatenate/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block1/ReluХ
%conv2d_1_block1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block1/Conv2D/ReadVariableOpю
conv2d_1_block1/Conv2DConv2Drelu0_block1/Relu:activations:0-conv2d_1_block1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block1/Conv2DМ
&conv2d_1_block1/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block1/BiasAdd/ReadVariableOpЪ
conv2d_1_block1/BiasAddBiasAddconv2d_1_block1/Conv2D:output:0.conv2d_1_block1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block1/BiasAdd
bn1_block1/ReadVariableOpReadVariableOp"bn1_block1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp
bn1_block1/ReadVariableOp_1ReadVariableOp$bn1_block1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block1/ReadVariableOp_1Ш
*bn1_block1/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block1/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block1/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block1/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block1/BiasAdd:output:0!bn1_block1/ReadVariableOp:value:0#bn1_block1/ReadVariableOp_1:value:02bn1_block1/FusedBatchNormV3/ReadVariableOp:value:04bn1_block1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block1/FusedBatchNormV3љ
bn1_block1/AssignNewValueAssignVariableOp3bn1_block1_fusedbatchnormv3_readvariableop_resource(bn1_block1/FusedBatchNormV3:batch_mean:0+^bn1_block1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block1/AssignNewValue
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
concatenate_1/concat/axisЬ
concatenate_1/concatConcatV2bn1_block1/FusedBatchNormV3:y:0inputs_2"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:џџџџџџџџџ2
concatenate_1/concat
relu1_block1/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block1/ReluЩ
max_pooling2d/MaxPoolMaxPoolrelu1_block1/Relu:activations:0*1
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolХ
%conv2d_0_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block2/Conv2D/ReadVariableOpэ
conv2d_0_block2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0-conv2d_0_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_0_block2/Conv2DМ
&conv2d_0_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block2/BiasAdd/ReadVariableOpЪ
conv2d_0_block2/BiasAddBiasAddconv2d_0_block2/Conv2D:output:0.conv2d_0_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_0_block2/BiasAdd
bn0_block2/ReadVariableOpReadVariableOp"bn0_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp
bn0_block2/ReadVariableOp_1ReadVariableOp$bn0_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block2/ReadVariableOp_1Ш
*bn0_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn0_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block2/BiasAdd:output:0!bn0_block2/ReadVariableOp:value:0#bn0_block2/ReadVariableOp_1:value:02bn0_block2/FusedBatchNormV3/ReadVariableOp:value:04bn0_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block2/FusedBatchNormV3љ
bn0_block2/AssignNewValueAssignVariableOp3bn0_block2_fusedbatchnormv3_readvariableop_resource(bn0_block2/FusedBatchNormV3:batch_mean:0+^bn0_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue
bn0_block2/AssignNewValue_1AssignVariableOp5bn0_block2_fusedbatchnormv3_readvariableop_1_resource,bn0_block2/FusedBatchNormV3:batch_variance:0-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block2/AssignNewValue_1
relu0_block2/ReluRelubn0_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu0_block2/ReluХ
%conv2d_1_block2/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block2/Conv2D/ReadVariableOpю
conv2d_1_block2/Conv2DConv2Drelu0_block2/Relu:activations:0-conv2d_1_block2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_1_block2/Conv2DМ
&conv2d_1_block2/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block2/BiasAdd/ReadVariableOpЪ
conv2d_1_block2/BiasAddBiasAddconv2d_1_block2/Conv2D:output:0.conv2d_1_block2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ2
conv2d_1_block2/BiasAdd
bn1_block2/ReadVariableOpReadVariableOp"bn1_block2_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp
bn1_block2/ReadVariableOp_1ReadVariableOp$bn1_block2_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block2/ReadVariableOp_1Ш
*bn1_block2/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block2/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block2/FusedBatchNormV3/ReadVariableOp_1Ж
bn1_block2/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block2/BiasAdd:output:0!bn1_block2/ReadVariableOp:value:0#bn1_block2/ReadVariableOp_1:value:02bn1_block2/FusedBatchNormV3/ReadVariableOp:value:04bn1_block2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block2/FusedBatchNormV3љ
bn1_block2/AssignNewValueAssignVariableOp3bn1_block2_fusedbatchnormv3_readvariableop_resource(bn1_block2/FusedBatchNormV3:batch_mean:0+^bn1_block2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue
bn1_block2/AssignNewValue_1AssignVariableOp5bn1_block2_fusedbatchnormv3_readvariableop_1_resource,bn1_block2/FusedBatchNormV3:batch_variance:0-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block2/AssignNewValue_1
relu1_block2/ReluRelubn1_block2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:џџџџџџџџџ2
relu1_block2/ReluЫ
max_pooling2d_1/MaxPoolMaxPoolrelu1_block2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolХ
%conv2d_0_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_0_block3/Conv2D/ReadVariableOpэ
conv2d_0_block3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0-conv2d_0_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_0_block3/Conv2DМ
&conv2d_0_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_0_block3/BiasAdd/ReadVariableOpШ
conv2d_0_block3/BiasAddBiasAddconv2d_0_block3/Conv2D:output:0.conv2d_0_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_0_block3/BiasAdd
bn0_block3/ReadVariableOpReadVariableOp"bn0_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp
bn0_block3/ReadVariableOp_1ReadVariableOp$bn0_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn0_block3/ReadVariableOp_1Ш
*bn0_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn0_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn0_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block3/BiasAdd:output:0!bn0_block3/ReadVariableOp:value:0#bn0_block3/ReadVariableOp_1:value:02bn0_block3/FusedBatchNormV3/ReadVariableOp:value:04bn0_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block3/FusedBatchNormV3љ
bn0_block3/AssignNewValueAssignVariableOp3bn0_block3_fusedbatchnormv3_readvariableop_resource(bn0_block3/FusedBatchNormV3:batch_mean:0+^bn0_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue
bn0_block3/AssignNewValue_1AssignVariableOp5bn0_block3_fusedbatchnormv3_readvariableop_1_resource,bn0_block3/FusedBatchNormV3:batch_variance:0-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block3/AssignNewValue_1
relu0_block3/ReluRelubn0_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu0_block3/ReluХ
%conv2d_1_block3/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%conv2d_1_block3/Conv2D/ReadVariableOpь
conv2d_1_block3/Conv2DConv2Drelu0_block3/Relu:activations:0-conv2d_1_block3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d_1_block3/Conv2DМ
&conv2d_1_block3/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&conv2d_1_block3/BiasAdd/ReadVariableOpШ
conv2d_1_block3/BiasAddBiasAddconv2d_1_block3/Conv2D:output:0.conv2d_1_block3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d_1_block3/BiasAdd
bn1_block3/ReadVariableOpReadVariableOp"bn1_block3_readvariableop_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp
bn1_block3/ReadVariableOp_1ReadVariableOp$bn1_block3_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1_block3/ReadVariableOp_1Ш
*bn1_block3/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*bn1_block3/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,bn1_block3/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block3/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block3/BiasAdd:output:0!bn1_block3/ReadVariableOp:value:0#bn1_block3/ReadVariableOp_1:value:02bn1_block3/FusedBatchNormV3/ReadVariableOp:value:04bn1_block3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block3/FusedBatchNormV3љ
bn1_block3/AssignNewValueAssignVariableOp3bn1_block3_fusedbatchnormv3_readvariableop_resource(bn1_block3/FusedBatchNormV3:batch_mean:0+^bn1_block3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue
bn1_block3/AssignNewValue_1AssignVariableOp5bn1_block3_fusedbatchnormv3_readvariableop_1_resource,bn1_block3/FusedBatchNormV3:batch_variance:0-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block3/AssignNewValue_1
relu1_block3/ReluRelubn1_block3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu1_block3/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpб
conv2d/Conv2DConv2Drelu1_block3/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
conv2d/BiasAddА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1с
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2&
$batch_normalization/FusedBatchNormV3І
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueВ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1
relu_C3_block3/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2
relu_C3_block3/ReluЭ
max_pooling2d_2/MaxPoolMaxPool!relu_C3_block3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolХ
%conv2d_0_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_0_block4_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02'
%conv2d_0_block4/Conv2D/ReadVariableOpэ
conv2d_0_block4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0-conv2d_0_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_0_block4/Conv2DМ
&conv2d_0_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_0_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_0_block4/BiasAdd/ReadVariableOpШ
conv2d_0_block4/BiasAddBiasAddconv2d_0_block4/Conv2D:output:0.conv2d_0_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_0_block4/BiasAdd
bn0_block4/ReadVariableOpReadVariableOp"bn0_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp
bn0_block4/ReadVariableOp_1ReadVariableOp$bn0_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn0_block4/ReadVariableOp_1Ш
*bn0_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn0_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn0_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn0_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_0_block4/BiasAdd:output:0!bn0_block4/ReadVariableOp:value:0#bn0_block4/ReadVariableOp_1:value:02bn0_block4/FusedBatchNormV3/ReadVariableOp:value:04bn0_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn0_block4/FusedBatchNormV3љ
bn0_block4/AssignNewValueAssignVariableOp3bn0_block4_fusedbatchnormv3_readvariableop_resource(bn0_block4/FusedBatchNormV3:batch_mean:0+^bn0_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue
bn0_block4/AssignNewValue_1AssignVariableOp5bn0_block4_fusedbatchnormv3_readvariableop_1_resource,bn0_block4/FusedBatchNormV3:batch_variance:0-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn0_block4/AssignNewValue_1
relu0_block4/ReluRelubn0_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu0_block4/ReluХ
%conv2d_1_block4/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_block4_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02'
%conv2d_1_block4/Conv2D/ReadVariableOpь
conv2d_1_block4/Conv2DConv2Drelu0_block4/Relu:activations:0-conv2d_1_block4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1_block4/Conv2DМ
&conv2d_1_block4/BiasAdd/ReadVariableOpReadVariableOp/conv2d_1_block4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&conv2d_1_block4/BiasAdd/ReadVariableOpШ
conv2d_1_block4/BiasAddBiasAddconv2d_1_block4/Conv2D:output:0.conv2d_1_block4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1_block4/BiasAdd
bn1_block4/ReadVariableOpReadVariableOp"bn1_block4_readvariableop_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp
bn1_block4/ReadVariableOp_1ReadVariableOp$bn1_block4_readvariableop_1_resource*
_output_shapes
:0*
dtype02
bn1_block4/ReadVariableOp_1Ш
*bn1_block4/FusedBatchNormV3/ReadVariableOpReadVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02,
*bn1_block4/FusedBatchNormV3/ReadVariableOpЮ
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02.
,bn1_block4/FusedBatchNormV3/ReadVariableOp_1Д
bn1_block4/FusedBatchNormV3FusedBatchNormV3 conv2d_1_block4/BiasAdd:output:0!bn1_block4/ReadVariableOp:value:0#bn1_block4/ReadVariableOp_1:value:02bn1_block4/FusedBatchNormV3/ReadVariableOp:value:04bn1_block4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2
bn1_block4/FusedBatchNormV3љ
bn1_block4/AssignNewValueAssignVariableOp3bn1_block4_fusedbatchnormv3_readvariableop_resource(bn1_block4/FusedBatchNormV3:batch_mean:0+^bn1_block4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue
bn1_block4/AssignNewValue_1AssignVariableOp5bn1_block4_fusedbatchnormv3_readvariableop_1_resource,bn1_block4/FusedBatchNormV3:batch_variance:0-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn1_block4/AssignNewValue_1
relu1_block4/ReluRelubn1_block4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu1_block4/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Drelu1_block4/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  0*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
conv2d_1/BiasAddЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1я
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<2(
&batch_normalization_1/FusedBatchNormV3А
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueМ
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
relu_C3_block4/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ  02
relu_C3_block4/ReluЭ
max_pooling2d_3/MaxPoolMaxPool!relu_C3_block4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolЉ
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesн
global_max_pooling2d/MaxMax max_pooling2d_3/MaxPool:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
	keep_dims(2
global_max_pooling2d/Max
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЪ
conv1d/conv1d/ExpandDims
ExpandDims!global_max_pooling2d/Max:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ02
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
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
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2Д
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      0   2
conv1d/conv1d/Reshape/shapeМ
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ02
conv1d/conv1d/Reshapeо
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
conv1d/conv1d/concat/axisи
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concatЙ
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ2
conv1d/conv1d/Reshape_1Е
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/ShapeЈ
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackЕ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1Ќ
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ќ
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_sliceЇ
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeй
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/Reshapeк
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpэ
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2#
!conv1d/squeeze_batch_dims/BiasAddЇ
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatц
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2%
#conv1d/squeeze_batch_dims/Reshape_1
conv1d/SigmoidSigmoid,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1d/SigmoidЋ
tf.compat.v1.squeeze/adj_outputSqueezeconv1d/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2!
tf.compat.v1.squeeze/adj_output
IdentityIdentity(tf.compat.v1.squeeze/adj_output:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЯ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^bn0_block1/AssignNewValue^bn0_block1/AssignNewValue_1+^bn0_block1/FusedBatchNormV3/ReadVariableOp-^bn0_block1/FusedBatchNormV3/ReadVariableOp_1^bn0_block1/ReadVariableOp^bn0_block1/ReadVariableOp_1^bn0_block2/AssignNewValue^bn0_block2/AssignNewValue_1+^bn0_block2/FusedBatchNormV3/ReadVariableOp-^bn0_block2/FusedBatchNormV3/ReadVariableOp_1^bn0_block2/ReadVariableOp^bn0_block2/ReadVariableOp_1^bn0_block3/AssignNewValue^bn0_block3/AssignNewValue_1+^bn0_block3/FusedBatchNormV3/ReadVariableOp-^bn0_block3/FusedBatchNormV3/ReadVariableOp_1^bn0_block3/ReadVariableOp^bn0_block3/ReadVariableOp_1^bn0_block4/AssignNewValue^bn0_block4/AssignNewValue_1+^bn0_block4/FusedBatchNormV3/ReadVariableOp-^bn0_block4/FusedBatchNormV3/ReadVariableOp_1^bn0_block4/ReadVariableOp^bn0_block4/ReadVariableOp_1^bn1_block1/AssignNewValue^bn1_block1/AssignNewValue_1+^bn1_block1/FusedBatchNormV3/ReadVariableOp-^bn1_block1/FusedBatchNormV3/ReadVariableOp_1^bn1_block1/ReadVariableOp^bn1_block1/ReadVariableOp_1^bn1_block2/AssignNewValue^bn1_block2/AssignNewValue_1+^bn1_block2/FusedBatchNormV3/ReadVariableOp-^bn1_block2/FusedBatchNormV3/ReadVariableOp_1^bn1_block2/ReadVariableOp^bn1_block2/ReadVariableOp_1^bn1_block3/AssignNewValue^bn1_block3/AssignNewValue_1+^bn1_block3/FusedBatchNormV3/ReadVariableOp-^bn1_block3/FusedBatchNormV3/ReadVariableOp_1^bn1_block3/ReadVariableOp^bn1_block3/ReadVariableOp_1^bn1_block4/AssignNewValue^bn1_block4/AssignNewValue_1+^bn1_block4/FusedBatchNormV3/ReadVariableOp-^bn1_block4/FusedBatchNormV3/ReadVariableOp_1^bn1_block4/ReadVariableOp^bn1_block4/ReadVariableOp_1*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp'^conv2d_0_block1/BiasAdd/ReadVariableOp&^conv2d_0_block1/Conv2D/ReadVariableOp'^conv2d_0_block2/BiasAdd/ReadVariableOp&^conv2d_0_block2/Conv2D/ReadVariableOp'^conv2d_0_block3/BiasAdd/ReadVariableOp&^conv2d_0_block3/Conv2D/ReadVariableOp'^conv2d_0_block4/BiasAdd/ReadVariableOp&^conv2d_0_block4/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp'^conv2d_1_block1/BiasAdd/ReadVariableOp&^conv2d_1_block1/Conv2D/ReadVariableOp'^conv2d_1_block2/BiasAdd/ReadVariableOp&^conv2d_1_block2/Conv2D/ReadVariableOp'^conv2d_1_block3/BiasAdd/ReadVariableOp&^conv2d_1_block3/Conv2D/ReadVariableOp'^conv2d_1_block4/BiasAdd/ReadVariableOp&^conv2d_1_block4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ш
_input_shapesж
г:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:џџџџџџџџџ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Ї
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7539

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Л
є
)__inference_bn1_block4_layer_call_fn_8748

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ02

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs


I__inference_conv2d_0_block3_layer_call_and_return_conditional_losses_7932

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Л
є
)__inference_bn1_block1_layer_call_fn_7456

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Љ
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8991

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж

D__inference_bn1_block1_layer_call_and_return_conditional_losses_7384

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"Ј-
saver_filename:0
Identity:0Identity_838"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*з
serving_defaultУ
I
	node_pair<
serving_default_node_pair:0џџџџџџџџџ
G
node_pos;
serving_default_node_pos:0џџџџџџџџџ
G
skel_img;
serving_default_skel_img:0џџџџџџџџџH
tf.compat.v1.squeeze0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Џі
ў

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
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ї
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Н

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+&call_and_return_all_conditional_losses
 __call__"
_tf_keras_layer
ь
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+Ё&call_and_return_all_conditional_losses
Ђ__call__"
_tf_keras_layer
Ї
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+Ѓ&call_and_return_all_conditional_losses
Є__call__"
_tf_keras_layer
Ї
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+Ѕ&call_and_return_all_conditional_losses
І__call__"
_tf_keras_layer
Н

Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+Ї&call_and_return_all_conditional_losses
Ј__call__"
_tf_keras_layer
ь
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+Љ&call_and_return_all_conditional_losses
Њ__call__"
_tf_keras_layer
Ї
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+Ћ&call_and_return_all_conditional_losses
Ќ__call__"
_tf_keras_layer
Ї
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
+­&call_and_return_all_conditional_losses
Ў__call__"
_tf_keras_layer
Ї
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+Џ&call_and_return_all_conditional_losses
А__call__"
_tf_keras_layer
Н

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"
_tf_keras_layer
ь
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"
_tf_keras_layer
Ї
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"
_tf_keras_layer
Р

}kernel
~bias
	variables
regularization_losses
trainable_variables
	keras_api
+З&call_and_return_all_conditional_losses
И__call__"
_tf_keras_layer
ѕ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"
_tf_keras_layer
Ћ
	variables
regularization_losses
trainable_variables
	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"
_tf_keras_layer
Ћ
	variables
regularization_losses
trainable_variables
	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"
_tf_keras_layer
У
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"
_tf_keras_layer
ѕ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"
_tf_keras_layer
Ћ
Ѓ	variables
Єregularization_losses
Ѕtrainable_variables
І	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"
_tf_keras_layer
У
Їkernel
	Јbias
Љ	variables
Њregularization_losses
Ћtrainable_variables
Ќ	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"
_tf_keras_layer
ѕ
	­axis

Ўgamma
	Џbeta
Аmoving_mean
Бmoving_variance
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"
_tf_keras_layer
Ћ
Ж	variables
Зregularization_losses
Иtrainable_variables
Й	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"
_tf_keras_layer
У
Кkernel
	Лbias
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"
_tf_keras_layer
ѕ
	Рaxis

Сgamma
	Тbeta
Уmoving_mean
Фmoving_variance
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"
_tf_keras_layer
Ћ
Щ	variables
Ъregularization_losses
Ыtrainable_variables
Ь	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"
_tf_keras_layer
Ћ
Э	variables
Юregularization_losses
Яtrainable_variables
а	keras_api
+б&call_and_return_all_conditional_losses
в__call__"
_tf_keras_layer
У
бkernel
	вbias
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
+г&call_and_return_all_conditional_losses
д__call__"
_tf_keras_layer
ѕ
	зaxis

иgamma
	йbeta
кmoving_mean
лmoving_variance
м	variables
нregularization_losses
оtrainable_variables
п	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layer
Ћ
р	variables
сregularization_losses
тtrainable_variables
у	keras_api
+з&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer
У
фkernel
	хbias
ц	variables
чregularization_losses
шtrainable_variables
щ	keras_api
+й&call_and_return_all_conditional_losses
к__call__"
_tf_keras_layer
ѕ
	ъaxis

ыgamma
	ьbeta
эmoving_mean
юmoving_variance
я	variables
№regularization_losses
ёtrainable_variables
ђ	keras_api
+л&call_and_return_all_conditional_losses
м__call__"
_tf_keras_layer
Ћ
ѓ	variables
єregularization_losses
ѕtrainable_variables
і	keras_api
+н&call_and_return_all_conditional_losses
о__call__"
_tf_keras_layer
У
їkernel
	јbias
љ	variables
њregularization_losses
ћtrainable_variables
ќ	keras_api
+п&call_and_return_all_conditional_losses
р__call__"
_tf_keras_layer
ѕ
	§axis

ўgamma
	џbeta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
+с&call_and_return_all_conditional_losses
т__call__"
_tf_keras_layer
Ћ
	variables
regularization_losses
trainable_variables
	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"
_tf_keras_layer
Ћ
	variables
regularization_losses
trainable_variables
	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"
_tf_keras_layer
Ћ
	variables
regularization_losses
trainable_variables
	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"
_tf_keras_layer
У
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
"
	optimizer
 "
trackable_list_wrapper
"
	optimizer
А
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
20
21
22
23
24
25
26
27
28
29
Ї30
Ј31
Ў32
Џ33
А34
Б35
К36
Л37
С38
Т39
У40
Ф41
б42
в43
и44
й45
к46
л47
ф48
х49
ы50
ь51
э52
ю53
ї54
ј55
ў56
џ57
58
59
60
61"
trackable_list_wrapper
 "
trackable_list_wrapper

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
14
15
16
17
18
19
Ї20
Ј21
Ў22
Џ23
К24
Л25
С26
Т27
б28
в29
и30
й31
ф32
х33
ы34
ь35
ї36
ј37
ў38
џ39
40
41"
trackable_list_wrapper
г
 layer_regularization_losses
layer_metrics
/	variables
metrics
non_trainable_variables
0regularization_losses
layers
1trainable_variables
ы__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
ьserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
layer_metrics
4	variables
 metrics
Ёnon_trainable_variables
5regularization_losses
Ђlayers
6trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
Е
 Ѓlayer_regularization_losses
Єlayer_metrics
:	variables
Ѕmetrics
Іnon_trainable_variables
;regularization_losses
Їlayers
<trainable_variables
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
Е
 Јlayer_regularization_losses
Љlayer_metrics
C	variables
Њmetrics
Ћnon_trainable_variables
Dregularization_losses
Ќlayers
Etrainable_variables
Ђ__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 ­layer_regularization_losses
Ўlayer_metrics
G	variables
Џmetrics
Аnon_trainable_variables
Hregularization_losses
Бlayers
Itrainable_variables
Є__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 Вlayer_regularization_losses
Гlayer_metrics
K	variables
Дmetrics
Еnon_trainable_variables
Lregularization_losses
Жlayers
Mtrainable_variables
І__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
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
Е
 Зlayer_regularization_losses
Иlayer_metrics
Q	variables
Йmetrics
Кnon_trainable_variables
Rregularization_losses
Лlayers
Strainable_variables
Ј__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
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
Е
 Мlayer_regularization_losses
Нlayer_metrics
Z	variables
Оmetrics
Пnon_trainable_variables
[regularization_losses
Рlayers
\trainable_variables
Њ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 Сlayer_regularization_losses
Тlayer_metrics
^	variables
Уmetrics
Фnon_trainable_variables
_regularization_losses
Хlayers
`trainable_variables
Ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 Цlayer_regularization_losses
Чlayer_metrics
b	variables
Шmetrics
Щnon_trainable_variables
cregularization_losses
Ъlayers
dtrainable_variables
Ў__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 Ыlayer_regularization_losses
Ьlayer_metrics
f	variables
Эmetrics
Юnon_trainable_variables
gregularization_losses
Яlayers
htrainable_variables
А__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
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
Е
 аlayer_regularization_losses
бlayer_metrics
l	variables
вmetrics
гnon_trainable_variables
mregularization_losses
дlayers
ntrainable_variables
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
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
Е
 еlayer_regularization_losses
жlayer_metrics
u	variables
зmetrics
иnon_trainable_variables
vregularization_losses
йlayers
wtrainable_variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 кlayer_regularization_losses
лlayer_metrics
y	variables
мmetrics
нnon_trainable_variables
zregularization_losses
оlayers
{trainable_variables
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
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
З
 пlayer_regularization_losses
рlayer_metrics
	variables
сmetrics
тnon_trainable_variables
regularization_losses
уlayers
trainable_variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn1_block2/gamma
:2bn1_block2/beta
&:$ (2bn1_block2/moving_mean
*:( (2bn1_block2/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
 фlayer_regularization_losses
хlayer_metrics
	variables
цmetrics
чnon_trainable_variables
regularization_losses
шlayers
trainable_variables
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 щlayer_regularization_losses
ъlayer_metrics
	variables
ыmetrics
ьnon_trainable_variables
regularization_losses
эlayers
trainable_variables
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 юlayer_regularization_losses
яlayer_metrics
	variables
№metrics
ёnon_trainable_variables
regularization_losses
ђlayers
trainable_variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_0_block3/kernel
": 2conv2d_0_block3/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
 ѓlayer_regularization_losses
єlayer_metrics
	variables
ѕmetrics
іnon_trainable_variables
regularization_losses
їlayers
trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn0_block3/gamma
:2bn0_block3/beta
&:$ (2bn0_block3/moving_mean
*:( (2bn0_block3/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
 јlayer_regularization_losses
љlayer_metrics
	variables
њmetrics
ћnon_trainable_variables
 regularization_losses
ќlayers
Ёtrainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 §layer_regularization_losses
ўlayer_metrics
Ѓ	variables
џmetrics
non_trainable_variables
Єregularization_losses
layers
Ѕtrainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
0:.2conv2d_1_block3/kernel
": 2conv2d_1_block3/bias
0
Ї0
Ј1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ї0
Ј1"
trackable_list_wrapper
И
 layer_regularization_losses
layer_metrics
Љ	variables
metrics
non_trainable_variables
Њregularization_losses
layers
Ћtrainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2bn1_block3/gamma
:2bn1_block3/beta
&:$ (2bn1_block3/moving_mean
*:( (2bn1_block3/moving_variance
@
Ў0
Џ1
А2
Б3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ў0
Џ1"
trackable_list_wrapper
И
 layer_regularization_losses
layer_metrics
В	variables
metrics
non_trainable_variables
Гregularization_losses
layers
Дtrainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 layer_regularization_losses
layer_metrics
Ж	variables
metrics
non_trainable_variables
Зregularization_losses
layers
Иtrainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
0
К0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
И
 layer_regularization_losses
layer_metrics
М	variables
metrics
non_trainable_variables
Нregularization_losses
layers
Оtrainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
@
С0
Т1
У2
Ф3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
И
 layer_regularization_losses
layer_metrics
Х	variables
metrics
non_trainable_variables
Цregularization_losses
layers
Чtrainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 layer_regularization_losses
layer_metrics
Щ	variables
metrics
non_trainable_variables
Ъregularization_losses
layers
Ыtrainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
  layer_regularization_losses
Ёlayer_metrics
Э	variables
Ђmetrics
Ѓnon_trainable_variables
Юregularization_losses
Єlayers
Яtrainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
0:.02conv2d_0_block4/kernel
": 02conv2d_0_block4/bias
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
И
 Ѕlayer_regularization_losses
Іlayer_metrics
г	variables
Їmetrics
Јnon_trainable_variables
дregularization_losses
Љlayers
еtrainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:02bn0_block4/gamma
:02bn0_block4/beta
&:$0 (2bn0_block4/moving_mean
*:(0 (2bn0_block4/moving_variance
@
и0
й1
к2
л3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
и0
й1"
trackable_list_wrapper
И
 Њlayer_regularization_losses
Ћlayer_metrics
м	variables
Ќmetrics
­non_trainable_variables
нregularization_losses
Ўlayers
оtrainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Џlayer_regularization_losses
Аlayer_metrics
р	variables
Бmetrics
Вnon_trainable_variables
сregularization_losses
Гlayers
тtrainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
0:.002conv2d_1_block4/kernel
": 02conv2d_1_block4/bias
0
ф0
х1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ф0
х1"
trackable_list_wrapper
И
 Дlayer_regularization_losses
Еlayer_metrics
ц	variables
Жmetrics
Зnon_trainable_variables
чregularization_losses
Иlayers
шtrainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:02bn1_block4/gamma
:02bn1_block4/beta
&:$0 (2bn1_block4/moving_mean
*:(0 (2bn1_block4/moving_variance
@
ы0
ь1
э2
ю3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ы0
ь1"
trackable_list_wrapper
И
 Йlayer_regularization_losses
Кlayer_metrics
я	variables
Лmetrics
Мnon_trainable_variables
№regularization_losses
Нlayers
ёtrainable_variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Оlayer_regularization_losses
Пlayer_metrics
ѓ	variables
Рmetrics
Сnon_trainable_variables
єregularization_losses
Тlayers
ѕtrainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
):'002conv2d_1/kernel
:02conv2d_1/bias
0
ї0
ј1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ї0
ј1"
trackable_list_wrapper
И
 Уlayer_regularization_losses
Фlayer_metrics
љ	variables
Хmetrics
Цnon_trainable_variables
њregularization_losses
Чlayers
ћtrainable_variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_1/gamma
(:&02batch_normalization_1/beta
1:/0 (2!batch_normalization_1/moving_mean
5:30 (2%batch_normalization_1/moving_variance
@
ў0
џ1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ў0
џ1"
trackable_list_wrapper
И
 Шlayer_regularization_losses
Щlayer_metrics
	variables
Ъmetrics
Ыnon_trainable_variables
regularization_losses
Ьlayers
trainable_variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Эlayer_regularization_losses
Юlayer_metrics
	variables
Яmetrics
аnon_trainable_variables
regularization_losses
бlayers
trainable_variables
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 вlayer_regularization_losses
гlayer_metrics
	variables
дmetrics
еnon_trainable_variables
regularization_losses
жlayers
trainable_variables
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 зlayer_regularization_losses
иlayer_metrics
	variables
йmetrics
кnon_trainable_variables
regularization_losses
лlayers
trainable_variables
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
#:!02conv1d/kernel
:2conv1d/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
 мlayer_regularization_losses
нlayer_metrics
	variables
оmetrics
пnon_trainable_variables
regularization_losses
рlayers
trainable_variables
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
p
с0
т1
у2
ф3
х4
ц5
ч6
ш7
щ8
ъ9"
trackable_list_wrapper
Ф
A0
B1
X2
Y3
s4
t5
6
7
8
9
А10
Б11
У12
Ф13
к14
л15
э16
ю17
18
19"
trackable_list_wrapper
ю
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
0
1"
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
0
1"
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
А0
Б1"
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
У0
Ф1"
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
к0
л1"
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
э0
ю1"
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
0
1"
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

ыtotal

ьcount
э	variables
ю	keras_api"
_tf_keras_metric
]
я
thresholds
№accumulator
ё	variables
ђ	keras_api"
_tf_keras_metric
]
ѓ
thresholds
єaccumulator
ѕ	variables
і	keras_api"
_tf_keras_metric
]
ї
thresholds
јaccumulator
љ	variables
њ	keras_api"
_tf_keras_metric
]
ћ
thresholds
ќaccumulator
§	variables
ў	keras_api"
_tf_keras_metric
c

џtotal

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
v

thresholds
true_positives
false_positives
	variables
	keras_api"
_tf_keras_metric
v

thresholds
true_positives
false_negatives
	variables
	keras_api"
_tf_keras_metric

true_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api"
_tf_keras_metric

true_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
ы0
ь1"
trackable_list_wrapper
.
э	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
№0"
trackable_list_wrapper
.
ё	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
є0"
trackable_list_wrapper
.
ѕ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
ј0"
trackable_list_wrapper
.
љ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
ќ0"
trackable_list_wrapper
.
§	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
џ0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:Ш (2true_positives
:Ш (2true_negatives
 :Ш (2false_positives
 :Ш (2false_negatives
@
0
1
2
3"
trackable_list_wrapper
.
	variables"
_generic_user_object
:Ш (2true_positives
:Ш (2true_negatives
 :Ш (2false_positives
 :Ш (2false_negatives
@
0
1
2
3"
trackable_list_wrapper
.
	variables"
_generic_user_object
Ю2Ы
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6853
@__inference_EdgeNN_layer_call_and_return_conditional_losses_7118
@__inference_EdgeNN_layer_call_and_return_conditional_losses_5889
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6154Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
рBн
__inference__wrapped_model_1194skel_imgnode_pos	node_pair"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
C__inference_summation_layer_call_and_return_conditional_losses_7128
C__inference_summation_layer_call_and_return_conditional_losses_7138Ц
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
 2
(__inference_summation_layer_call_fn_7148
(__inference_summation_layer_call_fn_7158Ц
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_0_block1_layer_call_and_return_conditional_losses_7168Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_0_block1_layer_call_fn_7178Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7196
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7214
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7232
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7250Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn0_block1_layer_call_fn_7268
)__inference_bn0_block1_layer_call_fn_7286
)__inference_bn0_block1_layer_call_fn_7304
)__inference_bn0_block1_layer_call_fn_7322Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
E__inference_concatenate_layer_call_and_return_conditional_losses_7329Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_concatenate_layer_call_fn_7336Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_relu0_block1_layer_call_and_return_conditional_losses_7341Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu0_block1_layer_call_fn_7346Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_1_block1_layer_call_and_return_conditional_losses_7356Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_1_block1_layer_call_fn_7366Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7384
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7402
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7420
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7438Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn1_block1_layer_call_fn_7456
)__inference_bn1_block1_layer_call_fn_7474
)__inference_bn1_block1_layer_call_fn_7492
)__inference_bn1_block1_layer_call_fn_7510Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ё2ю
G__inference_concatenate_1_layer_call_and_return_conditional_losses_7517Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_concatenate_1_layer_call_fn_7524Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_relu1_block1_layer_call_and_return_conditional_losses_7529Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu1_block1_layer_call_fn_7534Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
К2З
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7539
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7544Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
,__inference_max_pooling2d_layer_call_fn_7549
,__inference_max_pooling2d_layer_call_fn_7554Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_0_block2_layer_call_and_return_conditional_losses_7564Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_0_block2_layer_call_fn_7574Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7592
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7610
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7628
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7646Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn0_block2_layer_call_fn_7664
)__inference_bn0_block2_layer_call_fn_7682
)__inference_bn0_block2_layer_call_fn_7700
)__inference_bn0_block2_layer_call_fn_7718Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_relu0_block2_layer_call_and_return_conditional_losses_7723Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu0_block2_layer_call_fn_7728Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_1_block2_layer_call_and_return_conditional_losses_7738Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_1_block2_layer_call_fn_7748Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7766
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7784
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7802
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7820Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn1_block2_layer_call_fn_7838
)__inference_bn1_block2_layer_call_fn_7856
)__inference_bn1_block2_layer_call_fn_7874
)__inference_bn1_block2_layer_call_fn_7892Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_relu1_block2_layer_call_and_return_conditional_losses_7897Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu1_block2_layer_call_fn_7902Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
О2Л
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7907
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7912Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
.__inference_max_pooling2d_1_layer_call_fn_7917
.__inference_max_pooling2d_1_layer_call_fn_7922Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_0_block3_layer_call_and_return_conditional_losses_7932Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_0_block3_layer_call_fn_7942Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7960
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7978
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7996
D__inference_bn0_block3_layer_call_and_return_conditional_losses_8014Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn0_block3_layer_call_fn_8032
)__inference_bn0_block3_layer_call_fn_8050
)__inference_bn0_block3_layer_call_fn_8068
)__inference_bn0_block3_layer_call_fn_8086Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_relu0_block3_layer_call_and_return_conditional_losses_8091Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu0_block3_layer_call_fn_8096Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_1_block3_layer_call_and_return_conditional_losses_8106Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_1_block3_layer_call_fn_8116Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8134
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8152
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8170
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8188Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn1_block3_layer_call_fn_8206
)__inference_bn1_block3_layer_call_fn_8224
)__inference_bn1_block3_layer_call_fn_8242
)__inference_bn1_block3_layer_call_fn_8260Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_relu1_block3_layer_call_and_return_conditional_losses_8265Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu1_block3_layer_call_fn_8270Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_conv2d_layer_call_and_return_conditional_losses_8280Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_conv2d_layer_call_fn_8290Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8308
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8326
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8344
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8362Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
2__inference_batch_normalization_layer_call_fn_8380
2__inference_batch_normalization_layer_call_fn_8398
2__inference_batch_normalization_layer_call_fn_8416
2__inference_batch_normalization_layer_call_fn_8434Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
H__inference_relu_C3_block3_layer_call_and_return_conditional_losses_8439Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_relu_C3_block3_layer_call_fn_8444Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
О2Л
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8449
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8454Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
.__inference_max_pooling2d_2_layer_call_fn_8459
.__inference_max_pooling2d_2_layer_call_fn_8464Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_0_block4_layer_call_and_return_conditional_losses_8474Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_0_block4_layer_call_fn_8484Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8502
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8520
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8538
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8556Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn0_block4_layer_call_fn_8574
)__inference_bn0_block4_layer_call_fn_8592
)__inference_bn0_block4_layer_call_fn_8610
)__inference_bn0_block4_layer_call_fn_8628Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_relu0_block4_layer_call_and_return_conditional_losses_8633Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu0_block4_layer_call_fn_8638Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_conv2d_1_block4_layer_call_and_return_conditional_losses_8648Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_conv2d_1_block4_layer_call_fn_8658Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8676
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8694
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8712
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8730Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
)__inference_bn1_block4_layer_call_fn_8748
)__inference_bn1_block4_layer_call_fn_8766
)__inference_bn1_block4_layer_call_fn_8784
)__inference_bn1_block4_layer_call_fn_8802Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_relu1_block4_layer_call_and_return_conditional_losses_8807Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_relu1_block4_layer_call_fn_8812Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8822Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_conv2d_1_layer_call_fn_8832Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ў2ћ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8850
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8868
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8886
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8904Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
4__inference_batch_normalization_1_layer_call_fn_8922
4__inference_batch_normalization_1_layer_call_fn_8940
4__inference_batch_normalization_1_layer_call_fn_8958
4__inference_batch_normalization_1_layer_call_fn_8976Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
H__inference_relu_C3_block4_layer_call_and_return_conditional_losses_8981Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_relu_C3_block4_layer_call_fn_8986Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
О2Л
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8991
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8996Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
.__inference_max_pooling2d_3_layer_call_fn_9001
.__inference_max_pooling2d_3_layer_call_fn_9006Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9012
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9018Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
3__inference_global_max_pooling2d_layer_call_fn_9024
3__inference_global_max_pooling2d_layer_call_fn_9030Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_conv1d_layer_call_and_return_conditional_losses_9068Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_conv1d_layer_call_fn_9106Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
%__inference_EdgeNN_layer_call_fn_3034
%__inference_EdgeNN_layer_call_fn_9371
%__inference_EdgeNN_layer_call_fn_9636
%__inference_EdgeNN_layer_call_fn_5624Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
нBк
"__inference_signature_wrapper_6588	node_pairnode_posskel_img"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
@__inference_EdgeNN_layer_call_and_return_conditional_losses_5889Оh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЊЂІ
Ђ

,)
skel_imgџџџџџџџџџ
,)
node_posџџџџџџџџџ
-*
	node_pairџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6154Оh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЊЂІ
Ђ

,)
skel_imgџџџџџџџџџ
,)
node_posџџџџџџџџџ
-*
	node_pairџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
@__inference_EdgeNN_layer_call_and_return_conditional_losses_6853Нh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЉЂЅ
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
@__inference_EdgeNN_layer_call_and_return_conditional_losses_7118Нh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЉЂЅ
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 л
%__inference_EdgeNN_layer_call_fn_3034Бh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЊЂІ
Ђ

,)
skel_imgџџџџџџџџџ
,)
node_posџџџџџџџџџ
-*
	node_pairџџџџџџџџџ
p 

 
Њ "џџџџџџџџџл
%__inference_EdgeNN_layer_call_fn_5624Бh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЊЂІ
Ђ

,)
skel_imgџџџџџџџџџ
,)
node_posџџџџџџџџџ
-*
	node_pairџџџџџџџџџ
p

 
Њ "џџџџџџџџџк
%__inference_EdgeNN_layer_call_fn_9371Аh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЉЂЅ
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
p 

 
Њ "џџџџџџџџџк
%__inference_EdgeNN_layer_call_fn_9636Аh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЉЂЅ
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
__inference__wrapped_model_1194мh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџЂЂ
Ђ

,)
skel_imgџџџџџџџџџ
,)
node_posџџџџџџџџџ
-*
	node_pairџџџџџџџџџ
Њ "KЊH
F
tf.compat.v1.squeeze.+
tf.compat.v1.squeezeџџџџџџџџџю
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8850ўџMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 ю
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8868ўџMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 Щ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8886vўџ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p 
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 Щ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8904vўџ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 Ц
4__inference_batch_normalization_1_layer_call_fn_8922ўџMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0Ц
4__inference_batch_normalization_1_layer_call_fn_8940ўџMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0Ё
4__inference_batch_normalization_1_layer_call_fn_8958iўџ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p 
Њ " џџџџџџџџџ  0Ё
4__inference_batch_normalization_1_layer_call_fn_8976iўџ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p
Њ " џџџџџџџџџ  0ь
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8308СТУФMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8326СТУФMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8344vСТУФ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 Ч
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8362vСТУФ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 Ф
2__inference_batch_normalization_layer_call_fn_8380СТУФMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџФ
2__inference_batch_normalization_layer_call_fn_8398СТУФMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2__inference_batch_normalization_layer_call_fn_8416iСТУФ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ " џџџџџџџџџ@@
2__inference_batch_normalization_layer_call_fn_8434iСТУФ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ " џџџџџџџџџ@@п
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7196?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 п
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7214?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7232v?@AB=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ "/Ђ,
%"
0џџџџџџџџџ
 О
D__inference_bn0_block1_layer_call_and_return_conditional_losses_7250v?@AB=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ "/Ђ,
%"
0џџџџџџџџџ
 З
)__inference_bn0_block1_layer_call_fn_7268?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЗ
)__inference_bn0_block1_layer_call_fn_7286?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
)__inference_bn0_block1_layer_call_fn_7304i?@AB=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ ""џџџџџџџџџ
)__inference_bn0_block1_layer_call_fn_7322i?@AB=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ ""џџџџџџџџџп
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7592qrstMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 п
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7610qrstMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7628vqrst=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ "/Ђ,
%"
0џџџџџџџџџ
 О
D__inference_bn0_block2_layer_call_and_return_conditional_losses_7646vqrst=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ "/Ђ,
%"
0џџџџџџџџџ
 З
)__inference_bn0_block2_layer_call_fn_7664qrstMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЗ
)__inference_bn0_block2_layer_call_fn_7682qrstMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
)__inference_bn0_block2_layer_call_fn_7700iqrst=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ ""џџџџџџџџџ
)__inference_bn0_block2_layer_call_fn_7718iqrst=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ ""џџџџџџџџџу
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7960MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 у
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7978MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
D__inference_bn0_block3_layer_call_and_return_conditional_losses_7996v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 О
D__inference_bn0_block3_layer_call_and_return_conditional_losses_8014v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 Л
)__inference_bn0_block3_layer_call_fn_8032MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
)__inference_bn0_block3_layer_call_fn_8050MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
)__inference_bn0_block3_layer_call_fn_8068i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ " џџџџџџџџџ@@
)__inference_bn0_block3_layer_call_fn_8086i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ " џџџџџџџџџ@@у
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8502ийклMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 у
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8520ийклMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 О
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8538vийкл;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p 
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 О
D__inference_bn0_block4_layer_call_and_return_conditional_losses_8556vийкл;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 Л
)__inference_bn0_block4_layer_call_fn_8574ийклMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0Л
)__inference_bn0_block4_layer_call_fn_8592ийклMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
)__inference_bn0_block4_layer_call_fn_8610iийкл;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p 
Њ " џџџџџџџџџ  0
)__inference_bn0_block4_layer_call_fn_8628iийкл;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p
Њ " џџџџџџџџџ  0п
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7384VWXYMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 п
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7402VWXYMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7420vVWXY=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ "/Ђ,
%"
0џџџџџџџџџ
 О
D__inference_bn1_block1_layer_call_and_return_conditional_losses_7438vVWXY=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ "/Ђ,
%"
0џџџџџџџџџ
 З
)__inference_bn1_block1_layer_call_fn_7456VWXYMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЗ
)__inference_bn1_block1_layer_call_fn_7474VWXYMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
)__inference_bn1_block1_layer_call_fn_7492iVWXY=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ ""џџџџџџџџџ
)__inference_bn1_block1_layer_call_fn_7510iVWXY=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ ""џџџџџџџџџу
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7766MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 у
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7784MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7802z=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ "/Ђ,
%"
0џџџџџџџџџ
 Т
D__inference_bn1_block2_layer_call_and_return_conditional_losses_7820z=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ "/Ђ,
%"
0џџџџџџџџџ
 Л
)__inference_bn1_block2_layer_call_fn_7838MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
)__inference_bn1_block2_layer_call_fn_7856MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
)__inference_bn1_block2_layer_call_fn_7874m=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p 
Њ ""џџџџџџџџџ
)__inference_bn1_block2_layer_call_fn_7892m=Ђ:
3Ђ0
*'
inputsџџџџџџџџџ
p
Њ ""џџџџџџџџџу
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8134ЎЏАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 у
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8152ЎЏАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8170vЎЏАБ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 О
D__inference_bn1_block3_layer_call_and_return_conditional_losses_8188vЎЏАБ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 Л
)__inference_bn1_block3_layer_call_fn_8206ЎЏАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
)__inference_bn1_block3_layer_call_fn_8224ЎЏАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
)__inference_bn1_block3_layer_call_fn_8242iЎЏАБ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ " џџџџџџџџџ@@
)__inference_bn1_block3_layer_call_fn_8260iЎЏАБ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ " џџџџџџџџџ@@у
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8676ыьэюMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 у
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8694ыьэюMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 О
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8712vыьэю;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p 
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 О
D__inference_bn1_block4_layer_call_and_return_conditional_losses_8730vыьэю;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 Л
)__inference_bn1_block4_layer_call_fn_8748ыьэюMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0Л
)__inference_bn1_block4_layer_call_fn_8766ыьэюMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
)__inference_bn1_block4_layer_call_fn_8784iыьэю;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p 
Њ " џџџџџџџџџ  0
)__inference_bn1_block4_layer_call_fn_8802iыьэю;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ  0
p
Њ " џџџџџџџџџ  0э
G__inference_concatenate_1_layer_call_and_return_conditional_losses_7517ЁnЂk
dЂa
_\
,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 Х
,__inference_concatenate_1_layer_call_fn_7524nЂk
dЂa
_\
,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
Њ ""џџџџџџџџџы
E__inference_concatenate_layer_call_and_return_conditional_losses_7329ЁnЂk
dЂa
_\
,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 У
*__inference_concatenate_layer_call_fn_7336nЂk
dЂa
_\
,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
Њ ""џџџџџџџџџВ
@__inference_conv1d_layer_call_and_return_conditional_losses_9068n7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ0
Њ "-Ђ*
# 
0џџџџџџџџџ
 
%__inference_conv1d_layer_call_fn_9106a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ0
Њ " џџџџџџџџџН
I__inference_conv2d_0_block1_layer_call_and_return_conditional_losses_7168p899Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
.__inference_conv2d_0_block1_layer_call_fn_7178c899Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџН
I__inference_conv2d_0_block2_layer_call_and_return_conditional_losses_7564pjk9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
.__inference_conv2d_0_block2_layer_call_fn_7574cjk9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџЛ
I__inference_conv2d_0_block3_layer_call_and_return_conditional_losses_7932n7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
.__inference_conv2d_0_block3_layer_call_fn_7942a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ " џџџџџџџџџ@@Л
I__inference_conv2d_0_block4_layer_call_and_return_conditional_losses_8474nбв7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 
.__inference_conv2d_0_block4_layer_call_fn_8484aбв7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  
Њ " џџџџџџџџџ  0Н
I__inference_conv2d_1_block1_layer_call_and_return_conditional_losses_7356pOP9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
.__inference_conv2d_1_block1_layer_call_fn_7366cOP9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџН
I__inference_conv2d_1_block2_layer_call_and_return_conditional_losses_7738p}~9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
.__inference_conv2d_1_block2_layer_call_fn_7748c}~9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџЛ
I__inference_conv2d_1_block3_layer_call_and_return_conditional_losses_8106nЇЈ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
.__inference_conv2d_1_block3_layer_call_fn_8116aЇЈ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ " џџџџџџџџџ@@Л
I__inference_conv2d_1_block4_layer_call_and_return_conditional_losses_8648nфх7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 
.__inference_conv2d_1_block4_layer_call_fn_8658aфх7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ " џџџџџџџџџ  0Д
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8822nїј7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 
'__inference_conv2d_1_layer_call_fn_8832aїј7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ " џџџџџџџџџ  0В
@__inference_conv2d_layer_call_and_return_conditional_losses_8280nКЛ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
%__inference_conv2d_layer_call_fn_8290aКЛ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ " џџџџџџџџџ@@п
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9012RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "6Ђ3
,)
0"џџџџџџџџџџџџџџџџџџ
 К
N__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_9018h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ0
Њ "-Ђ*
# 
0џџџџџџџџџ0
 Ж
3__inference_global_max_pooling2d_layer_call_fn_9024RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ")&"џџџџџџџџџџџџџџџџџџ
3__inference_global_max_pooling2d_layer_call_fn_9030[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ0
Њ " џџџџџџџџџ0ь
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7907RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7912j9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 Ф
.__inference_max_pooling2d_1_layer_call_fn_7917RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
.__inference_max_pooling2d_1_layer_call_fn_7922]9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ " џџџџџџџџџ@@ь
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8449RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8454h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "-Ђ*
# 
0џџџџџџџџџ  
 Ф
.__inference_max_pooling2d_2_layer_call_fn_8459RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
.__inference_max_pooling2d_2_layer_call_fn_8464[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ " џџџџџџџџџ  ь
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8991RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8996h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ "-Ђ*
# 
0џџџџџџџџџ0
 Ф
.__inference_max_pooling2d_3_layer_call_fn_9001RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
.__inference_max_pooling2d_3_layer_call_fn_9006[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ " џџџџџџџџџ0ъ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7539RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7544l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 Т
,__inference_max_pooling2d_layer_call_fn_7549RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
,__inference_max_pooling2d_layer_call_fn_7554_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџЖ
F__inference_relu0_block1_layer_call_and_return_conditional_losses_7341l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
+__inference_relu0_block1_layer_call_fn_7346_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџЖ
F__inference_relu0_block2_layer_call_and_return_conditional_losses_7723l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
+__inference_relu0_block2_layer_call_fn_7728_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџВ
F__inference_relu0_block3_layer_call_and_return_conditional_losses_8091h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
+__inference_relu0_block3_layer_call_fn_8096[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ " џџџџџџџџџ@@В
F__inference_relu0_block4_layer_call_and_return_conditional_losses_8633h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 
+__inference_relu0_block4_layer_call_fn_8638[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ " џџџџџџџџџ  0Ж
F__inference_relu1_block1_layer_call_and_return_conditional_losses_7529l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
+__inference_relu1_block1_layer_call_fn_7534_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџЖ
F__inference_relu1_block2_layer_call_and_return_conditional_losses_7897l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџ
 
+__inference_relu1_block2_layer_call_fn_7902_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџВ
F__inference_relu1_block3_layer_call_and_return_conditional_losses_8265h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
+__inference_relu1_block3_layer_call_fn_8270[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ " џџџџџџџџџ@@В
F__inference_relu1_block4_layer_call_and_return_conditional_losses_8807h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 
+__inference_relu1_block4_layer_call_fn_8812[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ " џџџџџџџџџ  0Д
H__inference_relu_C3_block3_layer_call_and_return_conditional_losses_8439h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
-__inference_relu_C3_block3_layer_call_fn_8444[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ " џџџџџџџџџ@@Д
H__inference_relu_C3_block4_layer_call_and_return_conditional_losses_8981h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ "-Ђ*
# 
0џџџџџџџџџ  0
 
-__inference_relu_C3_block4_layer_call_fn_8986[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  0
Њ " џџџџџџџџџ  0Ё
"__inference_signature_wrapper_6588њh89?@ABOPVWXYjkqrst}~ЇЈЎЏАБКЛСТУФбвийклфхыьэюїјўџРЂМ
Ђ 
ДЊА
:
	node_pair-*
	node_pairџџџџџџџџџ
8
node_pos,)
node_posџџџџџџџџџ
8
skel_img,)
skel_imgџџџџџџџџџ"KЊH
F
tf.compat.v1.squeeze.+
tf.compat.v1.squeezeџџџџџџџџџ­
C__inference_summation_layer_call_and_return_conditional_losses_7128хБЂ­
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
Њ

trainingp "/Ђ,
%"
0џџџџџџџџџ
 ­
C__inference_summation_layer_call_and_return_conditional_losses_7138хБЂ­
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
Њ

trainingp"/Ђ,
%"
0џџџџџџџџџ
 
(__inference_summation_layer_call_fn_7148иБЂ­
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
Њ

trainingp ""џџџџџџџџџ
(__inference_summation_layer_call_fn_7158иБЂ­
Ђ

,)
inputs/0џџџџџџџџџ
,)
inputs/1џџџџџџџџџ
,)
inputs/2џџџџџџџџџ
Њ

trainingp""џџџџџџџџџ