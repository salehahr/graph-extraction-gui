??(
??
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
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.22unknown8??#
?
conv2d_0_relu_block_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameconv2d_0_relu_block_1/kernel
?
0conv2d_0_relu_block_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_relu_block_1/kernel*&
_output_shapes
:*
dtype0
?
conv2d_0_relu_block_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_0_relu_block_1/bias
?
.conv2d_0_relu_block_1/bias/Read/ReadVariableOpReadVariableOpconv2d_0_relu_block_1/bias*
_output_shapes
:*
dtype0
?
bn0_relu_block_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn0_relu_block_1/gamma
}
*bn0_relu_block_1/gamma/Read/ReadVariableOpReadVariableOpbn0_relu_block_1/gamma*
_output_shapes
:*
dtype0
?
bn0_relu_block_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namebn0_relu_block_1/beta
{
)bn0_relu_block_1/beta/Read/ReadVariableOpReadVariableOpbn0_relu_block_1/beta*
_output_shapes
:*
dtype0
?
bn0_relu_block_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebn0_relu_block_1/moving_mean
?
0bn0_relu_block_1/moving_mean/Read/ReadVariableOpReadVariableOpbn0_relu_block_1/moving_mean*
_output_shapes
:*
dtype0
?
 bn0_relu_block_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" bn0_relu_block_1/moving_variance
?
4bn0_relu_block_1/moving_variance/Read/ReadVariableOpReadVariableOp bn0_relu_block_1/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1_relu_block_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameconv2d_1_relu_block_1/kernel
?
0conv2d_1_relu_block_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_relu_block_1/kernel*&
_output_shapes
:*
dtype0
?
conv2d_1_relu_block_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_1_relu_block_1/bias
?
.conv2d_1_relu_block_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1_relu_block_1/bias*
_output_shapes
:*
dtype0
?
bn1_relu_block_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn1_relu_block_1/gamma
}
*bn1_relu_block_1/gamma/Read/ReadVariableOpReadVariableOpbn1_relu_block_1/gamma*
_output_shapes
:*
dtype0
?
bn1_relu_block_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namebn1_relu_block_1/beta
{
)bn1_relu_block_1/beta/Read/ReadVariableOpReadVariableOpbn1_relu_block_1/beta*
_output_shapes
:*
dtype0
?
bn1_relu_block_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebn1_relu_block_1/moving_mean
?
0bn1_relu_block_1/moving_mean/Read/ReadVariableOpReadVariableOpbn1_relu_block_1/moving_mean*
_output_shapes
:*
dtype0
?
 bn1_relu_block_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" bn1_relu_block_1/moving_variance
?
4bn1_relu_block_1/moving_variance/Read/ReadVariableOpReadVariableOp bn1_relu_block_1/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_0_relu_block_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameconv2d_0_relu_block_2/kernel
?
0conv2d_0_relu_block_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_relu_block_2/kernel*&
_output_shapes
: *
dtype0
?
conv2d_0_relu_block_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_0_relu_block_2/bias
?
.conv2d_0_relu_block_2/bias/Read/ReadVariableOpReadVariableOpconv2d_0_relu_block_2/bias*
_output_shapes
: *
dtype0
?
bn0_relu_block_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namebn0_relu_block_2/gamma
}
*bn0_relu_block_2/gamma/Read/ReadVariableOpReadVariableOpbn0_relu_block_2/gamma*
_output_shapes
: *
dtype0
?
bn0_relu_block_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namebn0_relu_block_2/beta
{
)bn0_relu_block_2/beta/Read/ReadVariableOpReadVariableOpbn0_relu_block_2/beta*
_output_shapes
: *
dtype0
?
bn0_relu_block_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebn0_relu_block_2/moving_mean
?
0bn0_relu_block_2/moving_mean/Read/ReadVariableOpReadVariableOpbn0_relu_block_2/moving_mean*
_output_shapes
: *
dtype0
?
 bn0_relu_block_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" bn0_relu_block_2/moving_variance
?
4bn0_relu_block_2/moving_variance/Read/ReadVariableOpReadVariableOp bn0_relu_block_2/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1_relu_block_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameconv2d_1_relu_block_2/kernel
?
0conv2d_1_relu_block_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_relu_block_2/kernel*&
_output_shapes
:  *
dtype0
?
conv2d_1_relu_block_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_1_relu_block_2/bias
?
.conv2d_1_relu_block_2/bias/Read/ReadVariableOpReadVariableOpconv2d_1_relu_block_2/bias*
_output_shapes
: *
dtype0
?
bn1_relu_block_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namebn1_relu_block_2/gamma
}
*bn1_relu_block_2/gamma/Read/ReadVariableOpReadVariableOpbn1_relu_block_2/gamma*
_output_shapes
: *
dtype0
?
bn1_relu_block_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namebn1_relu_block_2/beta
{
)bn1_relu_block_2/beta/Read/ReadVariableOpReadVariableOpbn1_relu_block_2/beta*
_output_shapes
: *
dtype0
?
bn1_relu_block_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebn1_relu_block_2/moving_mean
?
0bn1_relu_block_2/moving_mean/Read/ReadVariableOpReadVariableOpbn1_relu_block_2/moving_mean*
_output_shapes
: *
dtype0
?
 bn1_relu_block_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" bn1_relu_block_2/moving_variance
?
4bn1_relu_block_2/moving_variance/Read/ReadVariableOpReadVariableOp bn1_relu_block_2/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:*
dtype0
?
conv2d_0_relu_block_1r/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameconv2d_0_relu_block_1r/kernel
?
1conv2d_0_relu_block_1r/kernel/Read/ReadVariableOpReadVariableOpconv2d_0_relu_block_1r/kernel*&
_output_shapes
: *
dtype0
?
conv2d_0_relu_block_1r/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameconv2d_0_relu_block_1r/bias
?
/conv2d_0_relu_block_1r/bias/Read/ReadVariableOpReadVariableOpconv2d_0_relu_block_1r/bias*
_output_shapes
:*
dtype0
?
bn0_relu_block_1r/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebn0_relu_block_1r/gamma

+bn0_relu_block_1r/gamma/Read/ReadVariableOpReadVariableOpbn0_relu_block_1r/gamma*
_output_shapes
:*
dtype0
?
bn0_relu_block_1r/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn0_relu_block_1r/beta
}
*bn0_relu_block_1r/beta/Read/ReadVariableOpReadVariableOpbn0_relu_block_1r/beta*
_output_shapes
:*
dtype0
?
bn0_relu_block_1r/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebn0_relu_block_1r/moving_mean
?
1bn0_relu_block_1r/moving_mean/Read/ReadVariableOpReadVariableOpbn0_relu_block_1r/moving_mean*
_output_shapes
:*
dtype0
?
!bn0_relu_block_1r/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!bn0_relu_block_1r/moving_variance
?
5bn0_relu_block_1r/moving_variance/Read/ReadVariableOpReadVariableOp!bn0_relu_block_1r/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_1_relu_block_1r/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameconv2d_1_relu_block_1r/kernel
?
1conv2d_1_relu_block_1r/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_relu_block_1r/kernel*&
_output_shapes
:*
dtype0
?
conv2d_1_relu_block_1r/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameconv2d_1_relu_block_1r/bias
?
/conv2d_1_relu_block_1r/bias/Read/ReadVariableOpReadVariableOpconv2d_1_relu_block_1r/bias*
_output_shapes
:*
dtype0
?
bn1_relu_block_1r/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebn1_relu_block_1r/gamma

+bn1_relu_block_1r/gamma/Read/ReadVariableOpReadVariableOpbn1_relu_block_1r/gamma*
_output_shapes
:*
dtype0
?
bn1_relu_block_1r/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebn1_relu_block_1r/beta
}
*bn1_relu_block_1r/beta/Read/ReadVariableOpReadVariableOpbn1_relu_block_1r/beta*
_output_shapes
:*
dtype0
?
bn1_relu_block_1r/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebn1_relu_block_1r/moving_mean
?
1bn1_relu_block_1r/moving_mean/Read/ReadVariableOpReadVariableOpbn1_relu_block_1r/moving_mean*
_output_shapes
:*
dtype0
?
!bn1_relu_block_1r/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!bn1_relu_block_1r/moving_variance
?
5bn1_relu_block_1r/moving_variance/Read/ReadVariableOpReadVariableOp!bn1_relu_block_1r/moving_variance*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
d
total_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_11
]
total_11/Read/ReadVariableOpReadVariableOptotal_11*
_output_shapes
: *
dtype0
d
count_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_11
]
count_11/Read/ReadVariableOpReadVariableOpcount_11*
_output_shapes
: *
dtype0
d
total_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_12
]
total_12/Read/ReadVariableOpReadVariableOptotal_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0
d
total_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_13
]
total_13/Read/ReadVariableOpReadVariableOptotal_13*
_output_shapes
: *
dtype0
d
count_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_13
]
count_13/Read/ReadVariableOpReadVariableOpcount_13*
_output_shapes
: *
dtype0
d
total_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_14
]
total_14/Read/ReadVariableOpReadVariableOptotal_14*
_output_shapes
: *
dtype0
d
count_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_14
]
count_14/Read/ReadVariableOpReadVariableOpcount_14*
_output_shapes
: *
dtype0
d
total_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_15
]
total_15/Read/ReadVariableOpReadVariableOptotal_15*
_output_shapes
: *
dtype0
d
count_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_15
]
count_15/Read/ReadVariableOpReadVariableOpcount_15*
_output_shapes
: *
dtype0
d
total_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_16
]
total_16/Read/ReadVariableOpReadVariableOptotal_16*
_output_shapes
: *
dtype0
d
count_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_16
]
count_16/Read/ReadVariableOpReadVariableOpcount_16*
_output_shapes
: *
dtype0
d
total_17VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_17
]
total_17/Read/ReadVariableOpReadVariableOptotal_17*
_output_shapes
: *
dtype0
d
count_17VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_17
]
count_17/Read/ReadVariableOpReadVariableOpcount_17*
_output_shapes
: *
dtype0
d
total_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_18
]
total_18/Read/ReadVariableOpReadVariableOptotal_18*
_output_shapes
: *
dtype0
d
count_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_18
]
count_18/Read/ReadVariableOpReadVariableOpcount_18*
_output_shapes
: *
dtype0
d
total_19VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_19
]
total_19/Read/ReadVariableOpReadVariableOptotal_19*
_output_shapes
: *
dtype0
d
count_19VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_19
]
count_19/Read/ReadVariableOpReadVariableOpcount_19*
_output_shapes
: *
dtype0
d
total_20VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_20
]
total_20/Read/ReadVariableOpReadVariableOptotal_20*
_output_shapes
: *
dtype0
d
count_20VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_20
]
count_20/Read/ReadVariableOpReadVariableOpcount_20*
_output_shapes
: *
dtype0
d
total_21VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_21
]
total_21/Read/ReadVariableOpReadVariableOptotal_21*
_output_shapes
: *
dtype0
d
count_21VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_21
]
count_21/Read/ReadVariableOpReadVariableOpcount_21*
_output_shapes
: *
dtype0
d
total_22VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_22
]
total_22/Read/ReadVariableOpReadVariableOptotal_22*
_output_shapes
: *
dtype0
d
count_22VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_22
]
count_22/Read/ReadVariableOpReadVariableOpcount_22*
_output_shapes
: *
dtype0
d
total_23VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_23
]
total_23/Read/ReadVariableOpReadVariableOptotal_23*
_output_shapes
: *
dtype0
d
count_23VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_23
]
count_23/Read/ReadVariableOpReadVariableOpcount_23*
_output_shapes
: *
dtype0
d
total_24VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_24
]
total_24/Read/ReadVariableOpReadVariableOptotal_24*
_output_shapes
: *
dtype0
d
count_24VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_24
]
count_24/Read/ReadVariableOpReadVariableOpcount_24*
_output_shapes
: *
dtype0
d
total_25VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_25
]
total_25/Read/ReadVariableOpReadVariableOptotal_25*
_output_shapes
: *
dtype0
d
count_25VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_25
]
count_25/Read/ReadVariableOpReadVariableOpcount_25*
_output_shapes
: *
dtype0
d
total_26VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_26
]
total_26/Read/ReadVariableOpReadVariableOptotal_26*
_output_shapes
: *
dtype0
d
count_26VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_26
]
count_26/Read/ReadVariableOpReadVariableOpcount_26*
_output_shapes
: *
dtype0
d
total_27VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_27
]
total_27/Read/ReadVariableOpReadVariableOptotal_27*
_output_shapes
: *
dtype0
d
count_27VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_27
]
count_27/Read/ReadVariableOpReadVariableOpcount_27*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer_with_weights-12
layer-21
layer-22
layer_with_weights-13
layer-23
layer_with_weights-14
layer-24
layer_with_weights-15
layer-25
layer-26
layer-27
layer-28
	skips
	optimizer
 loss
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%
signatures
 
h

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
R
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
R
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
r	variables
strainable_variables
tregularization_losses
u	keras_api
R
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
h

zkernel
{bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
 
 
?
&0
'1
-2
.3
/4
05
96
:7
@8
A9
B10
C11
T12
U13
[14
\15
]16
^17
g18
h19
n20
o21
p22
q23
z24
{25
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
?
&0
'1
-2
.3
94
:5
@6
A7
T8
U9
[10
\11
g12
h13
n14
o15
z16
{17
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
 
?
!	variables
"trainable_variables
?layers
?metrics
#regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
hf
VARIABLE_VALUEconv2d_0_relu_block_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEconv2d_0_relu_block_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
?
(	variables
)trainable_variables
?layers
?metrics
*regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
a_
VARIABLE_VALUEbn0_relu_block_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbn0_relu_block_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn0_relu_block_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE bn0_relu_block_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
/2
03

-0
.1
 
?
1	variables
2trainable_variables
?layers
?metrics
3regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
5	variables
6trainable_variables
?layers
?metrics
7regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
hf
VARIABLE_VALUEconv2d_1_relu_block_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEconv2d_1_relu_block_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
;	variables
<trainable_variables
?layers
?metrics
=regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
a_
VARIABLE_VALUEbn1_relu_block_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbn1_relu_block_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn1_relu_block_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE bn1_relu_block_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
B2
C3

@0
A1
 
?
D	variables
Etrainable_variables
?layers
?metrics
Fregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
H	variables
Itrainable_variables
?layers
?metrics
Jregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
L	variables
Mtrainable_variables
?layers
?metrics
Nregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
P	variables
Qtrainable_variables
?layers
?metrics
Rregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
hf
VARIABLE_VALUEconv2d_0_relu_block_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEconv2d_0_relu_block_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
?
V	variables
Wtrainable_variables
?layers
?metrics
Xregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
a_
VARIABLE_VALUEbn0_relu_block_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbn0_relu_block_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn0_relu_block_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE bn0_relu_block_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
]2
^3

[0
\1
 
?
_	variables
`trainable_variables
?layers
?metrics
aregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
c	variables
dtrainable_variables
?layers
?metrics
eregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
hf
VARIABLE_VALUEconv2d_1_relu_block_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEconv2d_1_relu_block_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
i	variables
jtrainable_variables
?layers
?metrics
kregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
a_
VARIABLE_VALUEbn1_relu_block_2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbn1_relu_block_2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn1_relu_block_2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE bn1_relu_block_2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

n0
o1
p2
q3

n0
o1
 
?
r	variables
strainable_variables
?layers
?metrics
tregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
v	variables
wtrainable_variables
?layers
?metrics
xregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

z0
{1
 
?
|	variables
}trainable_variables
?layers
?metrics
~regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
ig
VARIABLE_VALUEconv2d_0_relu_block_1r/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEconv2d_0_relu_block_1r/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
ca
VARIABLE_VALUEbn0_relu_block_1r/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn0_relu_block_1r/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn0_relu_block_1r/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!bn0_relu_block_1r/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
jh
VARIABLE_VALUEconv2d_1_relu_block_1r/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEconv2d_1_relu_block_1r/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
ca
VARIABLE_VALUEbn1_relu_block_1r/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn1_relu_block_1r/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbn1_relu_block_1r/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!bn1_relu_block_1r/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
ZX
VARIABLE_VALUEconv2d/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_1/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_1/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_2/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_2/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?
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
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
 
 
Z
/0
01
B2
C3
]4
^5
p6
q7
?8
?9
?10
?11
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
/0
01
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
B0
C1
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
]0
^1
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
p0
q1
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

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_115keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_115keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_125keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_125keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_135keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_135keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_145keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_145keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_155keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_155keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_165keras_api/metrics/16/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_165keras_api/metrics/16/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_175keras_api/metrics/17/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_175keras_api/metrics/17/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_185keras_api/metrics/18/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_185keras_api/metrics/18/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_195keras_api/metrics/19/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_195keras_api/metrics/19/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_205keras_api/metrics/20/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_205keras_api/metrics/20/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_215keras_api/metrics/21/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_215keras_api/metrics/21/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_225keras_api/metrics/22/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_225keras_api/metrics/22/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_235keras_api/metrics/23/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_235keras_api/metrics/23/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_245keras_api/metrics/24/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_245keras_api/metrics/24/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_255keras_api/metrics/25/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_255keras_api/metrics/25/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_265keras_api/metrics/26/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_265keras_api/metrics/26/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_275keras_api/metrics/27/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_275keras_api/metrics/27/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
?
serving_default_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d_0_relu_block_1/kernelconv2d_0_relu_block_1/biasbn0_relu_block_1/gammabn0_relu_block_1/betabn0_relu_block_1/moving_mean bn0_relu_block_1/moving_varianceconv2d_1_relu_block_1/kernelconv2d_1_relu_block_1/biasbn1_relu_block_1/gammabn1_relu_block_1/betabn1_relu_block_1/moving_mean bn1_relu_block_1/moving_varianceconv2d_0_relu_block_2/kernelconv2d_0_relu_block_2/biasbn0_relu_block_2/gammabn0_relu_block_2/betabn0_relu_block_2/moving_mean bn0_relu_block_2/moving_varianceconv2d_1_relu_block_2/kernelconv2d_1_relu_block_2/biasbn1_relu_block_2/gammabn1_relu_block_2/betabn1_relu_block_2/moving_mean bn1_relu_block_2/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasconv2d_0_relu_block_1r/kernelconv2d_0_relu_block_1r/biasbn0_relu_block_1r/gammabn0_relu_block_1r/betabn0_relu_block_1r/moving_mean!bn0_relu_block_1r/moving_varianceconv2d_1_relu_block_1r/kernelconv2d_1_relu_block_1r/biasbn1_relu_block_1r/gammabn1_relu_block_1r/betabn1_relu_block_1r/moving_mean!bn1_relu_block_1r/moving_varianceconv2d_2/kernelconv2d_2/biasconv2d_1/kernelconv2d_1/biasconv2d/kernelconv2d/bias*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????:???????????:???????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_4833
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
?,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*?,
value?+B?+eB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/17/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/17/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/18/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/18/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/19/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/19/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/20/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/20/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/21/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/21/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/22/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/22/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/23/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/23/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/24/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/24/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/25/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/25/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/26/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/26/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/27/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/27/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*?
value?B?eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices0conv2d_0_relu_block_1/kernel/Read/ReadVariableOp.conv2d_0_relu_block_1/bias/Read/ReadVariableOp*bn0_relu_block_1/gamma/Read/ReadVariableOp)bn0_relu_block_1/beta/Read/ReadVariableOp0bn0_relu_block_1/moving_mean/Read/ReadVariableOp4bn0_relu_block_1/moving_variance/Read/ReadVariableOp0conv2d_1_relu_block_1/kernel/Read/ReadVariableOp.conv2d_1_relu_block_1/bias/Read/ReadVariableOp*bn1_relu_block_1/gamma/Read/ReadVariableOp)bn1_relu_block_1/beta/Read/ReadVariableOp0bn1_relu_block_1/moving_mean/Read/ReadVariableOp4bn1_relu_block_1/moving_variance/Read/ReadVariableOp0conv2d_0_relu_block_2/kernel/Read/ReadVariableOp.conv2d_0_relu_block_2/bias/Read/ReadVariableOp*bn0_relu_block_2/gamma/Read/ReadVariableOp)bn0_relu_block_2/beta/Read/ReadVariableOp0bn0_relu_block_2/moving_mean/Read/ReadVariableOp4bn0_relu_block_2/moving_variance/Read/ReadVariableOp0conv2d_1_relu_block_2/kernel/Read/ReadVariableOp.conv2d_1_relu_block_2/bias/Read/ReadVariableOp*bn1_relu_block_2/gamma/Read/ReadVariableOp)bn1_relu_block_2/beta/Read/ReadVariableOp0bn1_relu_block_2/moving_mean/Read/ReadVariableOp4bn1_relu_block_2/moving_variance/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp1conv2d_0_relu_block_1r/kernel/Read/ReadVariableOp/conv2d_0_relu_block_1r/bias/Read/ReadVariableOp+bn0_relu_block_1r/gamma/Read/ReadVariableOp*bn0_relu_block_1r/beta/Read/ReadVariableOp1bn0_relu_block_1r/moving_mean/Read/ReadVariableOp5bn0_relu_block_1r/moving_variance/Read/ReadVariableOp1conv2d_1_relu_block_1r/kernel/Read/ReadVariableOp/conv2d_1_relu_block_1r/bias/Read/ReadVariableOp+bn1_relu_block_1r/gamma/Read/ReadVariableOp*bn1_relu_block_1r/beta/Read/ReadVariableOp1bn1_relu_block_1r/moving_mean/Read/ReadVariableOp5bn1_relu_block_1r/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOptotal_11/Read/ReadVariableOpcount_11/Read/ReadVariableOptotal_12/Read/ReadVariableOpcount_12/Read/ReadVariableOptotal_13/Read/ReadVariableOpcount_13/Read/ReadVariableOptotal_14/Read/ReadVariableOpcount_14/Read/ReadVariableOptotal_15/Read/ReadVariableOpcount_15/Read/ReadVariableOptotal_16/Read/ReadVariableOpcount_16/Read/ReadVariableOptotal_17/Read/ReadVariableOpcount_17/Read/ReadVariableOptotal_18/Read/ReadVariableOpcount_18/Read/ReadVariableOptotal_19/Read/ReadVariableOpcount_19/Read/ReadVariableOptotal_20/Read/ReadVariableOpcount_20/Read/ReadVariableOptotal_21/Read/ReadVariableOpcount_21/Read/ReadVariableOptotal_22/Read/ReadVariableOpcount_22/Read/ReadVariableOptotal_23/Read/ReadVariableOpcount_23/Read/ReadVariableOptotal_24/Read/ReadVariableOpcount_24/Read/ReadVariableOptotal_25/Read/ReadVariableOpcount_25/Read/ReadVariableOptotal_26/Read/ReadVariableOpcount_26/Read/ReadVariableOptotal_27/Read/ReadVariableOpcount_27/Read/ReadVariableOpConst"/device:CPU:0*s
dtypesi
g2e
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
?,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*?,
value?+B?+eB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/17/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/17/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/18/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/18/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/19/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/19/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/20/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/20/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/21/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/21/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/22/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/22/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/23/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/23/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/24/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/24/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/25/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/25/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/26/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/26/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/27/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/27/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*?
value?B?eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*s
dtypesi
g2e
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOpAssignVariableOpconv2d_0_relu_block_1/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_1AssignVariableOpconv2d_0_relu_block_1/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_2AssignVariableOpbn0_relu_block_1/gamma
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_3AssignVariableOpbn0_relu_block_1/beta
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_4AssignVariableOpbn0_relu_block_1/moving_mean
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
p
AssignVariableOp_5AssignVariableOp bn0_relu_block_1/moving_variance
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_6AssignVariableOpconv2d_1_relu_block_1/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_7AssignVariableOpconv2d_1_relu_block_1/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_8AssignVariableOpbn1_relu_block_1/gamma
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_9AssignVariableOpbn1_relu_block_1/betaIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_10AssignVariableOpbn1_relu_block_1/moving_meanIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_11AssignVariableOp bn1_relu_block_1/moving_varianceIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_12AssignVariableOpconv2d_0_relu_block_2/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_13AssignVariableOpconv2d_0_relu_block_2/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_14AssignVariableOpbn0_relu_block_2/gammaIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_15AssignVariableOpbn0_relu_block_2/betaIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_16AssignVariableOpbn0_relu_block_2/moving_meanIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_17AssignVariableOp bn0_relu_block_2/moving_varianceIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_18AssignVariableOpconv2d_1_relu_block_2/kernelIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_19AssignVariableOpconv2d_1_relu_block_2/biasIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_20AssignVariableOpbn1_relu_block_2/gammaIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_21AssignVariableOpbn1_relu_block_2/betaIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_22AssignVariableOpbn1_relu_block_2/moving_meanIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_23AssignVariableOp bn1_relu_block_2/moving_varianceIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_24AssignVariableOpconv2d_transpose/kernelIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_25AssignVariableOpconv2d_transpose/biasIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_26AssignVariableOpconv2d_0_relu_block_1r/kernelIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_27AssignVariableOpconv2d_0_relu_block_1r/biasIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_28AssignVariableOpbn0_relu_block_1r/gammaIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_29AssignVariableOpbn0_relu_block_1r/betaIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_30AssignVariableOpbn0_relu_block_1r/moving_meanIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
s
AssignVariableOp_31AssignVariableOp!bn0_relu_block_1r/moving_varianceIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_32AssignVariableOpconv2d_1_relu_block_1r/kernelIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_33AssignVariableOpconv2d_1_relu_block_1r/biasIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_34AssignVariableOpbn1_relu_block_1r/gammaIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_35AssignVariableOpbn1_relu_block_1r/betaIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_36AssignVariableOpbn1_relu_block_1r/moving_meanIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
s
AssignVariableOp_37AssignVariableOp!bn1_relu_block_1r/moving_varianceIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_38AssignVariableOpconv2d/kernelIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_39AssignVariableOpconv2d/biasIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_40AssignVariableOpconv2d_1/kernelIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_41AssignVariableOpconv2d_1/biasIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_42AssignVariableOpconv2d_2/kernelIdentity_43"/device:CPU:0*
dtype0
W
Identity_44IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_43AssignVariableOpconv2d_2/biasIdentity_44"/device:CPU:0*
dtype0
W
Identity_45IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_44AssignVariableOptotalIdentity_45"/device:CPU:0*
dtype0
W
Identity_46IdentityRestoreV2:45"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_45AssignVariableOpcountIdentity_46"/device:CPU:0*
dtype0
W
Identity_47IdentityRestoreV2:46"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_46AssignVariableOptotal_1Identity_47"/device:CPU:0*
dtype0
W
Identity_48IdentityRestoreV2:47"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_47AssignVariableOpcount_1Identity_48"/device:CPU:0*
dtype0
W
Identity_49IdentityRestoreV2:48"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_48AssignVariableOptotal_2Identity_49"/device:CPU:0*
dtype0
W
Identity_50IdentityRestoreV2:49"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_49AssignVariableOpcount_2Identity_50"/device:CPU:0*
dtype0
W
Identity_51IdentityRestoreV2:50"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_50AssignVariableOptotal_3Identity_51"/device:CPU:0*
dtype0
W
Identity_52IdentityRestoreV2:51"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_51AssignVariableOpcount_3Identity_52"/device:CPU:0*
dtype0
W
Identity_53IdentityRestoreV2:52"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_52AssignVariableOptotal_4Identity_53"/device:CPU:0*
dtype0
W
Identity_54IdentityRestoreV2:53"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_53AssignVariableOpcount_4Identity_54"/device:CPU:0*
dtype0
W
Identity_55IdentityRestoreV2:54"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_54AssignVariableOptotal_5Identity_55"/device:CPU:0*
dtype0
W
Identity_56IdentityRestoreV2:55"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_55AssignVariableOpcount_5Identity_56"/device:CPU:0*
dtype0
W
Identity_57IdentityRestoreV2:56"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_56AssignVariableOptotal_6Identity_57"/device:CPU:0*
dtype0
W
Identity_58IdentityRestoreV2:57"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_57AssignVariableOpcount_6Identity_58"/device:CPU:0*
dtype0
W
Identity_59IdentityRestoreV2:58"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_58AssignVariableOptotal_7Identity_59"/device:CPU:0*
dtype0
W
Identity_60IdentityRestoreV2:59"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_59AssignVariableOpcount_7Identity_60"/device:CPU:0*
dtype0
W
Identity_61IdentityRestoreV2:60"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_60AssignVariableOptotal_8Identity_61"/device:CPU:0*
dtype0
W
Identity_62IdentityRestoreV2:61"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_61AssignVariableOpcount_8Identity_62"/device:CPU:0*
dtype0
W
Identity_63IdentityRestoreV2:62"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_62AssignVariableOptotal_9Identity_63"/device:CPU:0*
dtype0
W
Identity_64IdentityRestoreV2:63"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_63AssignVariableOpcount_9Identity_64"/device:CPU:0*
dtype0
W
Identity_65IdentityRestoreV2:64"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_64AssignVariableOptotal_10Identity_65"/device:CPU:0*
dtype0
W
Identity_66IdentityRestoreV2:65"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_65AssignVariableOpcount_10Identity_66"/device:CPU:0*
dtype0
W
Identity_67IdentityRestoreV2:66"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_66AssignVariableOptotal_11Identity_67"/device:CPU:0*
dtype0
W
Identity_68IdentityRestoreV2:67"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_67AssignVariableOpcount_11Identity_68"/device:CPU:0*
dtype0
W
Identity_69IdentityRestoreV2:68"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_68AssignVariableOptotal_12Identity_69"/device:CPU:0*
dtype0
W
Identity_70IdentityRestoreV2:69"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_69AssignVariableOpcount_12Identity_70"/device:CPU:0*
dtype0
W
Identity_71IdentityRestoreV2:70"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_70AssignVariableOptotal_13Identity_71"/device:CPU:0*
dtype0
W
Identity_72IdentityRestoreV2:71"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_71AssignVariableOpcount_13Identity_72"/device:CPU:0*
dtype0
W
Identity_73IdentityRestoreV2:72"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_72AssignVariableOptotal_14Identity_73"/device:CPU:0*
dtype0
W
Identity_74IdentityRestoreV2:73"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_73AssignVariableOpcount_14Identity_74"/device:CPU:0*
dtype0
W
Identity_75IdentityRestoreV2:74"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_74AssignVariableOptotal_15Identity_75"/device:CPU:0*
dtype0
W
Identity_76IdentityRestoreV2:75"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_75AssignVariableOpcount_15Identity_76"/device:CPU:0*
dtype0
W
Identity_77IdentityRestoreV2:76"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_76AssignVariableOptotal_16Identity_77"/device:CPU:0*
dtype0
W
Identity_78IdentityRestoreV2:77"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_77AssignVariableOpcount_16Identity_78"/device:CPU:0*
dtype0
W
Identity_79IdentityRestoreV2:78"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_78AssignVariableOptotal_17Identity_79"/device:CPU:0*
dtype0
W
Identity_80IdentityRestoreV2:79"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_79AssignVariableOpcount_17Identity_80"/device:CPU:0*
dtype0
W
Identity_81IdentityRestoreV2:80"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_80AssignVariableOptotal_18Identity_81"/device:CPU:0*
dtype0
W
Identity_82IdentityRestoreV2:81"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_81AssignVariableOpcount_18Identity_82"/device:CPU:0*
dtype0
W
Identity_83IdentityRestoreV2:82"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_82AssignVariableOptotal_19Identity_83"/device:CPU:0*
dtype0
W
Identity_84IdentityRestoreV2:83"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_83AssignVariableOpcount_19Identity_84"/device:CPU:0*
dtype0
W
Identity_85IdentityRestoreV2:84"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_84AssignVariableOptotal_20Identity_85"/device:CPU:0*
dtype0
W
Identity_86IdentityRestoreV2:85"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_85AssignVariableOpcount_20Identity_86"/device:CPU:0*
dtype0
W
Identity_87IdentityRestoreV2:86"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_86AssignVariableOptotal_21Identity_87"/device:CPU:0*
dtype0
W
Identity_88IdentityRestoreV2:87"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_87AssignVariableOpcount_21Identity_88"/device:CPU:0*
dtype0
W
Identity_89IdentityRestoreV2:88"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_88AssignVariableOptotal_22Identity_89"/device:CPU:0*
dtype0
W
Identity_90IdentityRestoreV2:89"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_89AssignVariableOpcount_22Identity_90"/device:CPU:0*
dtype0
W
Identity_91IdentityRestoreV2:90"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_90AssignVariableOptotal_23Identity_91"/device:CPU:0*
dtype0
W
Identity_92IdentityRestoreV2:91"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_91AssignVariableOpcount_23Identity_92"/device:CPU:0*
dtype0
W
Identity_93IdentityRestoreV2:92"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_92AssignVariableOptotal_24Identity_93"/device:CPU:0*
dtype0
W
Identity_94IdentityRestoreV2:93"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_93AssignVariableOpcount_24Identity_94"/device:CPU:0*
dtype0
W
Identity_95IdentityRestoreV2:94"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_94AssignVariableOptotal_25Identity_95"/device:CPU:0*
dtype0
W
Identity_96IdentityRestoreV2:95"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_95AssignVariableOpcount_25Identity_96"/device:CPU:0*
dtype0
W
Identity_97IdentityRestoreV2:96"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_96AssignVariableOptotal_26Identity_97"/device:CPU:0*
dtype0
W
Identity_98IdentityRestoreV2:97"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_97AssignVariableOpcount_26Identity_98"/device:CPU:0*
dtype0
W
Identity_99IdentityRestoreV2:98"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_98AssignVariableOptotal_27Identity_99"/device:CPU:0*
dtype0
X
Identity_100IdentityRestoreV2:99"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_99AssignVariableOpcount_27Identity_100"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_101Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??
?
h
L__inference_relu1_relu_block_1_layer_call_and_return_conditional_losses_5894

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5781

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_conv2d_1_relu_block_1_layer_call_and_return_conditional_losses_5735

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_5924

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
h
L__inference_relu1_relu_block_2_layer_call_and_return_conditional_losses_6296

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
??
?*
'__inference_nodes_nn_layer_call_fn_5551

inputsN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??bn0_relu_block_1/AssignNewValue?!bn0_relu_block_1/AssignNewValue_1?0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1? bn0_relu_block_1r/AssignNewValue?"bn0_relu_block_1r/AssignNewValue_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?bn0_relu_block_2/AssignNewValue?!bn0_relu_block_2/AssignNewValue_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?bn1_relu_block_1/AssignNewValue?!bn1_relu_block_1/AssignNewValue_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1? bn1_relu_block_1r/AssignNewValue?"bn1_relu_block_1r/AssignNewValue_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?bn1_relu_block_2/AssignNewValue?!bn1_relu_block_2/AssignNewValue_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinputs3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_1/FusedBatchNormV3?
bn0_relu_block_1/AssignNewValueAssignVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_1/FusedBatchNormV3:batch_mean:01^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_1/AssignNewValue?
!bn0_relu_block_1/AssignNewValue_1AssignVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_1/FusedBatchNormV3:batch_variance:03^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_1/AssignNewValue_1?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_1/FusedBatchNormV3?
bn1_relu_block_1/AssignNewValueAssignVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_1/FusedBatchNormV3:batch_mean:01^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_1/AssignNewValue?
!bn1_relu_block_1/AssignNewValue_1AssignVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_1/FusedBatchNormV3:batch_variance:03^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_1/AssignNewValue_1?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul_1?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/dropout/Mul_1:z:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_2/FusedBatchNormV3?
bn0_relu_block_2/AssignNewValueAssignVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_2/FusedBatchNormV3:batch_mean:01^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_2/AssignNewValue?
!bn0_relu_block_2/AssignNewValue_1AssignVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_2/FusedBatchNormV3:batch_variance:03^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_2/AssignNewValue_1?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_2/FusedBatchNormV3?
bn1_relu_block_2/AssignNewValueAssignVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_2/FusedBatchNormV3:batch_mean:01^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_2/AssignNewValue?
!bn1_relu_block_2/AssignNewValue_1AssignVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_2/FusedBatchNormV3:batch_variance:03^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_2/AssignNewValue_1?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn0_relu_block_1r/FusedBatchNormV3?
 bn0_relu_block_1r/AssignNewValueAssignVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn0_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn0_relu_block_1r/AssignNewValue?
"bn0_relu_block_1r/AssignNewValue_1AssignVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn0_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn0_relu_block_1r/AssignNewValue_1?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn1_relu_block_1r/FusedBatchNormV3?
 bn1_relu_block_1r/AssignNewValueAssignVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn1_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn1_relu_block_1r/AssignNewValue?
"bn1_relu_block_1r/AssignNewValue_1AssignVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn1_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn1_relu_block_1r/AssignNewValue_1?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp ^bn0_relu_block_1/AssignNewValue"^bn0_relu_block_1/AssignNewValue_11^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_1!^bn0_relu_block_1r/AssignNewValue#^bn0_relu_block_1r/AssignNewValue_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_1 ^bn0_relu_block_2/AssignNewValue"^bn0_relu_block_2/AssignNewValue_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_1 ^bn1_relu_block_1/AssignNewValue"^bn1_relu_block_1/AssignNewValue_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_1!^bn1_relu_block_1r/AssignNewValue#^bn1_relu_block_1r/AssignNewValue_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_1 ^bn1_relu_block_2/AssignNewValue"^bn1_relu_block_2/AssignNewValue_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
bn0_relu_block_1/AssignNewValuebn0_relu_block_1/AssignNewValue2F
!bn0_relu_block_1/AssignNewValue_1!bn0_relu_block_1/AssignNewValue_12d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12D
 bn0_relu_block_1r/AssignNewValue bn0_relu_block_1r/AssignNewValue2H
"bn0_relu_block_1r/AssignNewValue_1"bn0_relu_block_1r/AssignNewValue_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12B
bn0_relu_block_2/AssignNewValuebn0_relu_block_2/AssignNewValue2F
!bn0_relu_block_2/AssignNewValue_1!bn0_relu_block_2/AssignNewValue_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12B
bn1_relu_block_1/AssignNewValuebn1_relu_block_1/AssignNewValue2F
!bn1_relu_block_1/AssignNewValue_1!bn1_relu_block_1/AssignNewValue_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12D
 bn1_relu_block_1r/AssignNewValue bn1_relu_block_1r/AssignNewValue2H
"bn1_relu_block_1r/AssignNewValue_1"bn1_relu_block_1r/AssignNewValue_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12B
bn1_relu_block_2/AssignNewValuebn1_relu_block_2/AssignNewValue2F
!bn1_relu_block_2/AssignNewValue_1!bn1_relu_block_2/AssignNewValue_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
M
1__inference_relu1_relu_block_2_layer_call_fn_6301

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
0__inference_bn0_relu_block_1r_layer_call_fn_6537

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6501

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_0_relu_block_1_layer_call_fn_5571

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

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
?
?
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6165

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
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
0__inference_bn0_relu_block_1r_layer_call_fn_6555

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_bn1_relu_block_2_layer_call_fn_6273

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
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
0__inference_bn1_relu_block_1r_layer_call_fn_6729

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_relu0_relu_block_2_layer_call_fn_6127

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6009

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
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
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
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
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6045

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
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
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
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
/__inference_bn0_relu_block_2_layer_call_fn_6081

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
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
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
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
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5589

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_bn0_relu_block_1_layer_call_fn_5661

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6639

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_bn0_relu_block_1_layer_call_fn_5715

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_conv2d_0_relu_block_1r_layer_call_and_return_conditional_losses_6437

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
i
M__inference_relu1_relu_block_1r_layer_call_and_return_conditional_losses_6770

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5643

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?'
B__inference_nodes_nn_layer_call_and_return_conditional_losses_4320	
inputN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinput3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_1/FusedBatchNormV3?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_1/FusedBatchNormV3?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*1
_output_shapes
:???????????2
dropout/Identity?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/Identity:output:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_2/FusedBatchNormV3?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_2/FusedBatchNormV3?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn0_relu_block_1r/FusedBatchNormV3?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn1_relu_block_1r/FusedBatchNormV3?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp1^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
0__inference_bn0_relu_block_1r_layer_call_fn_6591

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5625

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6675

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6519

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_node_pos_layer_call_fn_6845

inputs
identitya
SigmoidSigmoidinputs*
T0*1
_output_shapes
:???????????2	
Sigmoidi
IdentityIdentitySigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_0_relu_block_1r_layer_call_fn_6447

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
`
D__inference_node_types_layer_call_and_return_conditional_losses_6860

inputs
identitya
SoftmaxSoftmaxinputs*
T0*1
_output_shapes
:???????????2	
Softmaxo
IdentityIdentitySoftmax:softmax:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn0_relu_block_2_layer_call_fn_6099

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5607

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
0__inference_bn1_relu_block_1r_layer_call_fn_6711

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_conv2d_1_relu_block_1r_layer_call_and_return_conditional_losses_6611

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
'__inference_conv2d_2_layer_call_fn_6835

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
N
2__inference_relu1_relu_block_1r_layer_call_fn_6775

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn1_relu_block_2_layer_call_fn_6255

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
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
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
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
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6805

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
M
1__inference_relu0_relu_block_1_layer_call_fn_5725

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_1_relu_block_1_layer_call_fn_5745

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
%__inference_conv2d_layer_call_fn_6795

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?*
'__inference_nodes_nn_layer_call_fn_4144	
inputN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??bn0_relu_block_1/AssignNewValue?!bn0_relu_block_1/AssignNewValue_1?0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1? bn0_relu_block_1r/AssignNewValue?"bn0_relu_block_1r/AssignNewValue_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?bn0_relu_block_2/AssignNewValue?!bn0_relu_block_2/AssignNewValue_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?bn1_relu_block_1/AssignNewValue?!bn1_relu_block_1/AssignNewValue_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1? bn1_relu_block_1r/AssignNewValue?"bn1_relu_block_1r/AssignNewValue_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?bn1_relu_block_2/AssignNewValue?!bn1_relu_block_2/AssignNewValue_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinput3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_1/FusedBatchNormV3?
bn0_relu_block_1/AssignNewValueAssignVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_1/FusedBatchNormV3:batch_mean:01^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_1/AssignNewValue?
!bn0_relu_block_1/AssignNewValue_1AssignVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_1/FusedBatchNormV3:batch_variance:03^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_1/AssignNewValue_1?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_1/FusedBatchNormV3?
bn1_relu_block_1/AssignNewValueAssignVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_1/FusedBatchNormV3:batch_mean:01^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_1/AssignNewValue?
!bn1_relu_block_1/AssignNewValue_1AssignVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_1/FusedBatchNormV3:batch_variance:03^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_1/AssignNewValue_1?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul_1?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/dropout/Mul_1:z:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_2/FusedBatchNormV3?
bn0_relu_block_2/AssignNewValueAssignVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_2/FusedBatchNormV3:batch_mean:01^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_2/AssignNewValue?
!bn0_relu_block_2/AssignNewValue_1AssignVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_2/FusedBatchNormV3:batch_variance:03^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_2/AssignNewValue_1?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_2/FusedBatchNormV3?
bn1_relu_block_2/AssignNewValueAssignVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_2/FusedBatchNormV3:batch_mean:01^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_2/AssignNewValue?
!bn1_relu_block_2/AssignNewValue_1AssignVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_2/FusedBatchNormV3:batch_variance:03^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_2/AssignNewValue_1?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn0_relu_block_1r/FusedBatchNormV3?
 bn0_relu_block_1r/AssignNewValueAssignVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn0_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn0_relu_block_1r/AssignNewValue?
"bn0_relu_block_1r/AssignNewValue_1AssignVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn0_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn0_relu_block_1r/AssignNewValue_1?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn1_relu_block_1r/FusedBatchNormV3?
 bn1_relu_block_1r/AssignNewValueAssignVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn1_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn1_relu_block_1r/AssignNewValue?
"bn1_relu_block_1r/AssignNewValue_1AssignVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn1_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn1_relu_block_1r/AssignNewValue_1?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp ^bn0_relu_block_1/AssignNewValue"^bn0_relu_block_1/AssignNewValue_11^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_1!^bn0_relu_block_1r/AssignNewValue#^bn0_relu_block_1r/AssignNewValue_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_1 ^bn0_relu_block_2/AssignNewValue"^bn0_relu_block_2/AssignNewValue_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_1 ^bn1_relu_block_1/AssignNewValue"^bn1_relu_block_1/AssignNewValue_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_1!^bn1_relu_block_1r/AssignNewValue#^bn1_relu_block_1r/AssignNewValue_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_1 ^bn1_relu_block_2/AssignNewValue"^bn1_relu_block_2/AssignNewValue_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
bn0_relu_block_1/AssignNewValuebn0_relu_block_1/AssignNewValue2F
!bn0_relu_block_1/AssignNewValue_1!bn0_relu_block_1/AssignNewValue_12d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12D
 bn0_relu_block_1r/AssignNewValue bn0_relu_block_1r/AssignNewValue2H
"bn0_relu_block_1r/AssignNewValue_1"bn0_relu_block_1r/AssignNewValue_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12B
bn0_relu_block_2/AssignNewValuebn0_relu_block_2/AssignNewValue2F
!bn0_relu_block_2/AssignNewValue_1!bn0_relu_block_2/AssignNewValue_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12B
bn1_relu_block_1/AssignNewValuebn1_relu_block_1/AssignNewValue2F
!bn1_relu_block_1/AssignNewValue_1!bn1_relu_block_1/AssignNewValue_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12D
 bn1_relu_block_1r/AssignNewValue bn1_relu_block_1r/AssignNewValue2H
"bn1_relu_block_1r/AssignNewValue_1"bn1_relu_block_1r/AssignNewValue_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12B
bn1_relu_block_2/AssignNewValuebn1_relu_block_2/AssignNewValue2F
!bn1_relu_block_2/AssignNewValue_1!bn1_relu_block_2/AssignNewValue_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?

?
'__inference_conv2d_1_layer_call_fn_6815

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?'
'__inference_nodes_nn_layer_call_fn_5368

inputsN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinputs3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_1/FusedBatchNormV3?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_1/FusedBatchNormV3?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*1
_output_shapes
:???????????2
dropout/Identity?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/Identity:output:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_2/FusedBatchNormV3?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_2/FusedBatchNormV3?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn0_relu_block_1r/FusedBatchNormV3?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn1_relu_block_1r/FusedBatchNormV3?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp1^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
V
*__inference_concatenate_layer_call_fn_6427
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
:??????????? 2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
D
&__inference_dropout_layer_call_fn_5941

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn0_relu_block_2_layer_call_fn_6063

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
/__inference_bn1_relu_block_1_layer_call_fn_5871

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_1_relu_block_2_layer_call_fn_6147

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6357

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
5__inference_conv2d_1_relu_block_1r_layer_call_fn_6621

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn1_relu_block_2_layer_call_fn_6291

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
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
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
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
?
?
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6201

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
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
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5817

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?'
B__inference_nodes_nn_layer_call_and_return_conditional_losses_5009

inputsN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinputs3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_1/FusedBatchNormV3?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_1/FusedBatchNormV3?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*1
_output_shapes
:???????????2
dropout/Identity?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/Identity:output:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_2/FusedBatchNormV3?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_2/FusedBatchNormV3?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn0_relu_block_1r/FusedBatchNormV3?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn1_relu_block_1r/FusedBatchNormV3?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp1^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn1_relu_block_1_layer_call_fn_5889

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_conv2d_1_relu_block_2_layer_call_and_return_conditional_losses_6137

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5763

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_relu1_relu_block_1_layer_call_fn_5899

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
i
M__inference_relu0_relu_block_1r_layer_call_and_return_conditional_losses_6596

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5799

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6219

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
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
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
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
?
?
0__inference_bn0_relu_block_1r_layer_call_fn_6573

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
0__inference_bn1_relu_block_1r_layer_call_fn_6765

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5904

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
?
E
&__inference_dropout_layer_call_fn_5953

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?%
?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6334

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6825

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
N
2__inference_relu0_relu_block_1r_layer_call_fn_6601

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?'
'__inference_nodes_nn_layer_call_fn_2396	
inputN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinput3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_1/FusedBatchNormV3?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_1/FusedBatchNormV3?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*1
_output_shapes
:???????????2
dropout/Identity?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/Identity:output:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn0_relu_block_2/FusedBatchNormV3?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2#
!bn1_relu_block_2/FusedBatchNormV3?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn0_relu_block_1r/FusedBatchNormV3?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2$
"bn1_relu_block_1r/FusedBatchNormV3?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp1^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?$
?
/__inference_conv2d_transpose_layer_call_fn_6390

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
0__inference_bn1_relu_block_1r_layer_call_fn_6747

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn1_relu_block_1_layer_call_fn_5835

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?*
B__inference_nodes_nn_layer_call_and_return_conditional_losses_4503	
inputN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??bn0_relu_block_1/AssignNewValue?!bn0_relu_block_1/AssignNewValue_1?0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1? bn0_relu_block_1r/AssignNewValue?"bn0_relu_block_1r/AssignNewValue_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?bn0_relu_block_2/AssignNewValue?!bn0_relu_block_2/AssignNewValue_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?bn1_relu_block_1/AssignNewValue?!bn1_relu_block_1/AssignNewValue_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1? bn1_relu_block_1r/AssignNewValue?"bn1_relu_block_1r/AssignNewValue_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?bn1_relu_block_2/AssignNewValue?!bn1_relu_block_2/AssignNewValue_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinput3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_1/FusedBatchNormV3?
bn0_relu_block_1/AssignNewValueAssignVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_1/FusedBatchNormV3:batch_mean:01^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_1/AssignNewValue?
!bn0_relu_block_1/AssignNewValue_1AssignVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_1/FusedBatchNormV3:batch_variance:03^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_1/AssignNewValue_1?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_1/FusedBatchNormV3?
bn1_relu_block_1/AssignNewValueAssignVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_1/FusedBatchNormV3:batch_mean:01^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_1/AssignNewValue?
!bn1_relu_block_1/AssignNewValue_1AssignVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_1/FusedBatchNormV3:batch_variance:03^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_1/AssignNewValue_1?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul_1?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/dropout/Mul_1:z:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_2/FusedBatchNormV3?
bn0_relu_block_2/AssignNewValueAssignVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_2/FusedBatchNormV3:batch_mean:01^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_2/AssignNewValue?
!bn0_relu_block_2/AssignNewValue_1AssignVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_2/FusedBatchNormV3:batch_variance:03^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_2/AssignNewValue_1?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_2/FusedBatchNormV3?
bn1_relu_block_2/AssignNewValueAssignVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_2/FusedBatchNormV3:batch_mean:01^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_2/AssignNewValue?
!bn1_relu_block_2/AssignNewValue_1AssignVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_2/FusedBatchNormV3:batch_variance:03^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_2/AssignNewValue_1?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn0_relu_block_1r/FusedBatchNormV3?
 bn0_relu_block_1r/AssignNewValueAssignVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn0_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn0_relu_block_1r/AssignNewValue?
"bn0_relu_block_1r/AssignNewValue_1AssignVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn0_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn0_relu_block_1r/AssignNewValue_1?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn1_relu_block_1r/FusedBatchNormV3?
 bn1_relu_block_1r/AssignNewValueAssignVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn1_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn1_relu_block_1r/AssignNewValue?
"bn1_relu_block_1r/AssignNewValue_1AssignVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn1_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn1_relu_block_1r/AssignNewValue_1?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp ^bn0_relu_block_1/AssignNewValue"^bn0_relu_block_1/AssignNewValue_11^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_1!^bn0_relu_block_1r/AssignNewValue#^bn0_relu_block_1r/AssignNewValue_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_1 ^bn0_relu_block_2/AssignNewValue"^bn0_relu_block_2/AssignNewValue_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_1 ^bn1_relu_block_1/AssignNewValue"^bn1_relu_block_1/AssignNewValue_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_1!^bn1_relu_block_1r/AssignNewValue#^bn1_relu_block_1r/AssignNewValue_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_1 ^bn1_relu_block_2/AssignNewValue"^bn1_relu_block_2/AssignNewValue_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
bn0_relu_block_1/AssignNewValuebn0_relu_block_1/AssignNewValue2F
!bn0_relu_block_1/AssignNewValue_1!bn0_relu_block_1/AssignNewValue_12d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12D
 bn0_relu_block_1r/AssignNewValue bn0_relu_block_1r/AssignNewValue2H
"bn0_relu_block_1r/AssignNewValue_1"bn0_relu_block_1r/AssignNewValue_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12B
bn0_relu_block_2/AssignNewValuebn0_relu_block_2/AssignNewValue2F
!bn0_relu_block_2/AssignNewValue_1!bn0_relu_block_2/AssignNewValue_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12B
bn1_relu_block_1/AssignNewValuebn1_relu_block_1/AssignNewValue2F
!bn1_relu_block_1/AssignNewValue_1!bn1_relu_block_1/AssignNewValue_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12D
 bn1_relu_block_1r/AssignNewValue bn1_relu_block_1r/AssignNewValue2H
"bn1_relu_block_1r/AssignNewValue_1"bn1_relu_block_1r/AssignNewValue_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12B
bn1_relu_block_2/AssignNewValuebn1_relu_block_2/AssignNewValue2F
!bn1_relu_block_2/AssignNewValue_1!bn1_relu_block_2/AssignNewValue_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_5991

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_conv2d_0_relu_block_1_layer_call_and_return_conditional_losses_5561

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

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
?
?
/__inference_bn0_relu_block_1_layer_call_fn_5697

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6183

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
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
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
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
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6693

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
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
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
^
B__inference_node_pos_layer_call_and_return_conditional_losses_6840

inputs
identitya
SigmoidSigmoidinputs*
T0*1
_output_shapes
:???????????2	
Sigmoidi
IdentityIdentitySigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6483

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_max_pooling2d_layer_call_fn_5914

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
/__inference_bn1_relu_block_2_layer_call_fn_6237

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
E
)__inference_node_types_layer_call_fn_6865

inputs
identitya
SoftmaxSoftmaxinputs*
T0*1
_output_shapes
:???????????2	
Softmaxo
IdentityIdentitySoftmax:softmax:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn1_relu_block_1_layer_call_fn_5853

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?*
B__inference_nodes_nn_layer_call_and_return_conditional_losses_5192

inputsN
4conv2d_0_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_0_relu_block_1_biasadd_readvariableop_resource:6
(bn0_relu_block_1_readvariableop_resource:8
*bn0_relu_block_1_readvariableop_1_resource:G
9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_1_relu_block_1_conv2d_readvariableop_resource:C
5conv2d_1_relu_block_1_biasadd_readvariableop_resource:6
(bn1_relu_block_1_readvariableop_resource:8
*bn1_relu_block_1_readvariableop_1_resource:G
9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:I
;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:N
4conv2d_0_relu_block_2_conv2d_readvariableop_resource: C
5conv2d_0_relu_block_2_biasadd_readvariableop_resource: 6
(bn0_relu_block_2_readvariableop_resource: 8
*bn0_relu_block_2_readvariableop_1_resource: G
9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: N
4conv2d_1_relu_block_2_conv2d_readvariableop_resource:  C
5conv2d_1_relu_block_2_biasadd_readvariableop_resource: 6
(bn1_relu_block_2_readvariableop_resource: 8
*bn1_relu_block_2_readvariableop_1_resource: G
9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: I
;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:O
5conv2d_0_relu_block_1r_conv2d_readvariableop_resource: D
6conv2d_0_relu_block_1r_biasadd_readvariableop_resource:7
)bn0_relu_block_1r_readvariableop_resource:9
+bn0_relu_block_1r_readvariableop_1_resource:H
:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:O
5conv2d_1_relu_block_1r_conv2d_readvariableop_resource:D
6conv2d_1_relu_block_1r_biasadd_readvariableop_resource:7
)bn1_relu_block_1r_readvariableop_resource:9
+bn1_relu_block_1r_readvariableop_1_resource:H
:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:J
<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??bn0_relu_block_1/AssignNewValue?!bn0_relu_block_1/AssignNewValue_1?0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_1/ReadVariableOp?!bn0_relu_block_1/ReadVariableOp_1? bn0_relu_block_1r/AssignNewValue?"bn0_relu_block_1r/AssignNewValue_1?1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn0_relu_block_1r/ReadVariableOp?"bn0_relu_block_1r/ReadVariableOp_1?bn0_relu_block_2/AssignNewValue?!bn0_relu_block_2/AssignNewValue_1?0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn0_relu_block_2/ReadVariableOp?!bn0_relu_block_2/ReadVariableOp_1?bn1_relu_block_1/AssignNewValue?!bn1_relu_block_1/AssignNewValue_1?0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_1/ReadVariableOp?!bn1_relu_block_1/ReadVariableOp_1? bn1_relu_block_1r/AssignNewValue?"bn1_relu_block_1r/AssignNewValue_1?1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1? bn1_relu_block_1r/ReadVariableOp?"bn1_relu_block_1r/ReadVariableOp_1?bn1_relu_block_2/AssignNewValue?!bn1_relu_block_2/AssignNewValue_1?0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?bn1_relu_block_2/ReadVariableOp?!bn1_relu_block_2/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
+conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1/Conv2DConv2Dinputs3conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1/Conv2D?
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1/BiasAddBiasAdd%conv2d_0_relu_block_1/Conv2D:output:04conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_0_relu_block_1/BiasAdd?
bn0_relu_block_1/ReadVariableOpReadVariableOp(bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn0_relu_block_1/ReadVariableOp?
!bn0_relu_block_1/ReadVariableOp_1ReadVariableOp*bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn0_relu_block_1/ReadVariableOp_1?
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_1/BiasAdd:output:0'bn0_relu_block_1/ReadVariableOp:value:0)bn0_relu_block_1/ReadVariableOp_1:value:08bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_1/FusedBatchNormV3?
bn0_relu_block_1/AssignNewValueAssignVariableOp9bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_1/FusedBatchNormV3:batch_mean:01^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_1/AssignNewValue?
!bn0_relu_block_1/AssignNewValue_1AssignVariableOp;bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_1/FusedBatchNormV3:batch_variance:03^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_1/AssignNewValue_1?
relu0_relu_block_1/ReluRelu%bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1/Relu?
+conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1/Conv2DConv2D%relu0_relu_block_1/Relu:activations:03conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1/Conv2D?
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1/BiasAddBiasAdd%conv2d_1_relu_block_1/Conv2D:output:04conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1_relu_block_1/BiasAdd?
bn1_relu_block_1/ReadVariableOpReadVariableOp(bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02!
bn1_relu_block_1/ReadVariableOp?
!bn1_relu_block_1/ReadVariableOp_1ReadVariableOp*bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!bn1_relu_block_1/ReadVariableOp_1?
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_1/BiasAdd:output:0'bn1_relu_block_1/ReadVariableOp:value:0)bn1_relu_block_1/ReadVariableOp_1:value:08bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_1/FusedBatchNormV3?
bn1_relu_block_1/AssignNewValueAssignVariableOp9bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_1/FusedBatchNormV3:batch_mean:01^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_1/AssignNewValue?
!bn1_relu_block_1/AssignNewValue_1AssignVariableOp;bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_1/FusedBatchNormV3:batch_variance:03^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_1/AssignNewValue_1?
relu1_relu_block_1/ReluRelu%bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1/Relu?
max_pooling2d/MaxPoolMaxPool%relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/dropout/Mul_1?
+conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
conv2d_0_relu_block_2/Conv2DConv2Ddropout/dropout/Mul_1:z:03conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_0_relu_block_2/Conv2D?
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_2/BiasAddBiasAdd%conv2d_0_relu_block_2/Conv2D:output:04conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_0_relu_block_2/BiasAdd?
bn0_relu_block_2/ReadVariableOpReadVariableOp(bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn0_relu_block_2/ReadVariableOp?
!bn0_relu_block_2/ReadVariableOp_1ReadVariableOp*bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn0_relu_block_2/ReadVariableOp_1?
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_0_relu_block_2/BiasAdd:output:0'bn0_relu_block_2/ReadVariableOp:value:0)bn0_relu_block_2/ReadVariableOp_1:value:08bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn0_relu_block_2/FusedBatchNormV3?
bn0_relu_block_2/AssignNewValueAssignVariableOp9bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn0_relu_block_2/FusedBatchNormV3:batch_mean:01^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn0_relu_block_2/AssignNewValue?
!bn0_relu_block_2/AssignNewValue_1AssignVariableOp;bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn0_relu_block_2/FusedBatchNormV3:batch_variance:03^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn0_relu_block_2/AssignNewValue_1?
relu0_relu_block_2/ReluRelu%bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu0_relu_block_2/Relu?
+conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp4conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
conv2d_1_relu_block_2/Conv2DConv2D%relu0_relu_block_2/Relu:activations:03conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_1_relu_block_2/Conv2D?
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp5conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_2/BiasAddBiasAdd%conv2d_1_relu_block_2/Conv2D:output:04conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_1_relu_block_2/BiasAdd?
bn1_relu_block_2/ReadVariableOpReadVariableOp(bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02!
bn1_relu_block_2/ReadVariableOp?
!bn1_relu_block_2/ReadVariableOp_1ReadVariableOp*bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!bn1_relu_block_2/ReadVariableOp_1?
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
!bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3&conv2d_1_relu_block_2/BiasAdd:output:0'bn1_relu_block_2/ReadVariableOp:value:0)bn1_relu_block_2/ReadVariableOp_1:value:08bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0:bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2#
!bn1_relu_block_2/FusedBatchNormV3?
bn1_relu_block_2/AssignNewValueAssignVariableOp9bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource.bn1_relu_block_2/FusedBatchNormV3:batch_mean:01^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02!
bn1_relu_block_2/AssignNewValue?
!bn1_relu_block_2/AssignNewValue_1AssignVariableOp;bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource2bn1_relu_block_2/FusedBatchNormV3:batch_variance:03^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02#
!bn1_relu_block_2/AssignNewValue_1?
relu1_relu_block_2/ReluRelu%bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
relu1_relu_block_2/Relu?
conv2d_transpose/ShapeShape%relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2%relu1_relu_block_1/Relu:activations:0!conv2d_transpose/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
concatenate/concat?
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_0_relu_block_1r/Conv2DConv2Dconcatenate/concat:output:04conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_0_relu_block_1r/Conv2D?
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_0_relu_block_1r/BiasAddBiasAdd&conv2d_0_relu_block_1r/Conv2D:output:05conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_0_relu_block_1r/BiasAdd?
 bn0_relu_block_1r/ReadVariableOpReadVariableOp)bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn0_relu_block_1r/ReadVariableOp?
"bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn0_relu_block_1r/ReadVariableOp_1?
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_0_relu_block_1r/BiasAdd:output:0(bn0_relu_block_1r/ReadVariableOp:value:0*bn0_relu_block_1r/ReadVariableOp_1:value:09bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn0_relu_block_1r/FusedBatchNormV3?
 bn0_relu_block_1r/AssignNewValueAssignVariableOp:bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn0_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn0_relu_block_1r/AssignNewValue?
"bn0_relu_block_1r/AssignNewValue_1AssignVariableOp<bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn0_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn0_relu_block_1r/AssignNewValue_1?
relu0_relu_block_1r/ReluRelu&bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu0_relu_block_1r/Relu?
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp5conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
conv2d_1_relu_block_1r/Conv2DConv2D&relu0_relu_block_1r/Relu:activations:04conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1_relu_block_1r/Conv2D?
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
conv2d_1_relu_block_1r/BiasAddBiasAdd&conv2d_1_relu_block_1r/Conv2D:output:05conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
conv2d_1_relu_block_1r/BiasAdd?
 bn1_relu_block_1r/ReadVariableOpReadVariableOp)bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02"
 bn1_relu_block_1r/ReadVariableOp?
"bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp+bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"bn1_relu_block_1r/ReadVariableOp_1?
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
"bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV3'conv2d_1_relu_block_1r/BiasAdd:output:0(bn1_relu_block_1r/ReadVariableOp:value:0*bn1_relu_block_1r/ReadVariableOp_1:value:09bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0;bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2$
"bn1_relu_block_1r/FusedBatchNormV3?
 bn1_relu_block_1r/AssignNewValueAssignVariableOp:bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource/bn1_relu_block_1r/FusedBatchNormV3:batch_mean:02^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 bn1_relu_block_1r/AssignNewValue?
"bn1_relu_block_1r/AssignNewValue_1AssignVariableOp<bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource3bn1_relu_block_1r/FusedBatchNormV3:batch_variance:04^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"bn1_relu_block_1r/AssignNewValue_1?
relu1_relu_block_1r/ReluRelu&bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
relu1_relu_block_1r/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_2/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D&relu1_relu_block_1r/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
node_types/SoftmaxSoftmaxconv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_types/Softmax?
degrees/SoftmaxSoftmaxconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
degrees/Softmax?
node_pos/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
node_pos/Sigmoidy
IdentityIdentitynode_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitydegrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identitynode_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp ^bn0_relu_block_1/AssignNewValue"^bn0_relu_block_1/AssignNewValue_11^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_1/ReadVariableOp"^bn0_relu_block_1/ReadVariableOp_1!^bn0_relu_block_1r/AssignNewValue#^bn0_relu_block_1r/AssignNewValue_12^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn0_relu_block_1r/ReadVariableOp#^bn0_relu_block_1r/ReadVariableOp_1 ^bn0_relu_block_2/AssignNewValue"^bn0_relu_block_2/AssignNewValue_11^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn0_relu_block_2/ReadVariableOp"^bn0_relu_block_2/ReadVariableOp_1 ^bn1_relu_block_1/AssignNewValue"^bn1_relu_block_1/AssignNewValue_11^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_1/ReadVariableOp"^bn1_relu_block_1/ReadVariableOp_1!^bn1_relu_block_1r/AssignNewValue#^bn1_relu_block_1r/AssignNewValue_12^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp4^bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1!^bn1_relu_block_1r/ReadVariableOp#^bn1_relu_block_1r/ReadVariableOp_1 ^bn1_relu_block_2/AssignNewValue"^bn1_relu_block_2/AssignNewValue_11^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp3^bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1 ^bn1_relu_block_2/ReadVariableOp"^bn1_relu_block_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_1/Conv2D/ReadVariableOp.^conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_0_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_0_relu_block_2/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp-^conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_1/Conv2D/ReadVariableOp.^conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-^conv2d_1_relu_block_1r/Conv2D/ReadVariableOp-^conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,^conv2d_1_relu_block_2/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
bn0_relu_block_1/AssignNewValuebn0_relu_block_1/AssignNewValue2F
!bn0_relu_block_1/AssignNewValue_1!bn0_relu_block_1/AssignNewValue_12d
0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_1/ReadVariableOpbn0_relu_block_1/ReadVariableOp2F
!bn0_relu_block_1/ReadVariableOp_1!bn0_relu_block_1/ReadVariableOp_12D
 bn0_relu_block_1r/AssignNewValue bn0_relu_block_1r/AssignNewValue2H
"bn0_relu_block_1r/AssignNewValue_1"bn0_relu_block_1r/AssignNewValue_12f
1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn0_relu_block_1r/ReadVariableOp bn0_relu_block_1r/ReadVariableOp2H
"bn0_relu_block_1r/ReadVariableOp_1"bn0_relu_block_1r/ReadVariableOp_12B
bn0_relu_block_2/AssignNewValuebn0_relu_block_2/AssignNewValue2F
!bn0_relu_block_2/AssignNewValue_1!bn0_relu_block_2/AssignNewValue_12d
0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn0_relu_block_2/ReadVariableOpbn0_relu_block_2/ReadVariableOp2F
!bn0_relu_block_2/ReadVariableOp_1!bn0_relu_block_2/ReadVariableOp_12B
bn1_relu_block_1/AssignNewValuebn1_relu_block_1/AssignNewValue2F
!bn1_relu_block_1/AssignNewValue_1!bn1_relu_block_1/AssignNewValue_12d
0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_1/ReadVariableOpbn1_relu_block_1/ReadVariableOp2F
!bn1_relu_block_1/ReadVariableOp_1!bn1_relu_block_1/ReadVariableOp_12D
 bn1_relu_block_1r/AssignNewValue bn1_relu_block_1r/AssignNewValue2H
"bn1_relu_block_1r/AssignNewValue_1"bn1_relu_block_1r/AssignNewValue_12f
1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp1bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2j
3bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_13bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12D
 bn1_relu_block_1r/ReadVariableOp bn1_relu_block_1r/ReadVariableOp2H
"bn1_relu_block_1r/ReadVariableOp_1"bn1_relu_block_1r/ReadVariableOp_12B
bn1_relu_block_2/AssignNewValuebn1_relu_block_2/AssignNewValue2F
!bn1_relu_block_2/AssignNewValue_1!bn1_relu_block_2/AssignNewValue_12d
0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp0bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2h
2bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12B
bn1_relu_block_2/ReadVariableOpbn1_relu_block_2/ReadVariableOp2F
!bn1_relu_block_2/ReadVariableOp_1!bn1_relu_block_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp,conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_1/Conv2D/ReadVariableOp+conv2d_0_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp,conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp,conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_0_relu_block_2/Conv2D/ReadVariableOp+conv2d_0_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp,conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_1/Conv2D/ReadVariableOp+conv2d_1_relu_block_1/Conv2D/ReadVariableOp2^
-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp-conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2\
,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp,conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2\
,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp,conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2Z
+conv2d_1_relu_block_2/Conv2D/ReadVariableOp+conv2d_1_relu_block_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_bn0_relu_block_2_layer_call_fn_6117

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
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
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
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
?
q
E__inference_concatenate_layer_call_and_return_conditional_losses_6420
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
:??????????? 2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
H
,__inference_max_pooling2d_layer_call_fn_5919

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
У
?-
__inference__wrapped_model_1187	
inputW
=nodes_nn_conv2d_0_relu_block_1_conv2d_readvariableop_resource:L
>nodes_nn_conv2d_0_relu_block_1_biasadd_readvariableop_resource:?
1nodes_nn_bn0_relu_block_1_readvariableop_resource:A
3nodes_nn_bn0_relu_block_1_readvariableop_1_resource:P
Bnodes_nn_bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource:R
Dnodes_nn_bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:W
=nodes_nn_conv2d_1_relu_block_1_conv2d_readvariableop_resource:L
>nodes_nn_conv2d_1_relu_block_1_biasadd_readvariableop_resource:?
1nodes_nn_bn1_relu_block_1_readvariableop_resource:A
3nodes_nn_bn1_relu_block_1_readvariableop_1_resource:P
Bnodes_nn_bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource:R
Dnodes_nn_bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource:W
=nodes_nn_conv2d_0_relu_block_2_conv2d_readvariableop_resource: L
>nodes_nn_conv2d_0_relu_block_2_biasadd_readvariableop_resource: ?
1nodes_nn_bn0_relu_block_2_readvariableop_resource: A
3nodes_nn_bn0_relu_block_2_readvariableop_1_resource: P
Bnodes_nn_bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource: R
Dnodes_nn_bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: W
=nodes_nn_conv2d_1_relu_block_2_conv2d_readvariableop_resource:  L
>nodes_nn_conv2d_1_relu_block_2_biasadd_readvariableop_resource: ?
1nodes_nn_bn1_relu_block_2_readvariableop_resource: A
3nodes_nn_bn1_relu_block_2_readvariableop_1_resource: P
Bnodes_nn_bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource: R
Dnodes_nn_bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource: \
Bnodes_nn_conv2d_transpose_conv2d_transpose_readvariableop_resource: G
9nodes_nn_conv2d_transpose_biasadd_readvariableop_resource:X
>nodes_nn_conv2d_0_relu_block_1r_conv2d_readvariableop_resource: M
?nodes_nn_conv2d_0_relu_block_1r_biasadd_readvariableop_resource:@
2nodes_nn_bn0_relu_block_1r_readvariableop_resource:B
4nodes_nn_bn0_relu_block_1r_readvariableop_1_resource:Q
Cnodes_nn_bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource:S
Enodes_nn_bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:X
>nodes_nn_conv2d_1_relu_block_1r_conv2d_readvariableop_resource:M
?nodes_nn_conv2d_1_relu_block_1r_biasadd_readvariableop_resource:@
2nodes_nn_bn1_relu_block_1r_readvariableop_resource:B
4nodes_nn_bn1_relu_block_1r_readvariableop_1_resource:Q
Cnodes_nn_bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource:S
Enodes_nn_bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource:J
0nodes_nn_conv2d_2_conv2d_readvariableop_resource:?
1nodes_nn_conv2d_2_biasadd_readvariableop_resource:J
0nodes_nn_conv2d_1_conv2d_readvariableop_resource:?
1nodes_nn_conv2d_1_biasadd_readvariableop_resource:H
.nodes_nn_conv2d_conv2d_readvariableop_resource:=
/nodes_nn_conv2d_biasadd_readvariableop_resource:
identity

identity_1

identity_2??9nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?;nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?(nodes_nn/bn0_relu_block_1/ReadVariableOp?*nodes_nn/bn0_relu_block_1/ReadVariableOp_1?:nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?<nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?)nodes_nn/bn0_relu_block_1r/ReadVariableOp?+nodes_nn/bn0_relu_block_1r/ReadVariableOp_1?9nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?;nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?(nodes_nn/bn0_relu_block_2/ReadVariableOp?*nodes_nn/bn0_relu_block_2/ReadVariableOp_1?9nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?;nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?(nodes_nn/bn1_relu_block_1/ReadVariableOp?*nodes_nn/bn1_relu_block_1/ReadVariableOp_1?:nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?<nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?)nodes_nn/bn1_relu_block_1r/ReadVariableOp?+nodes_nn/bn1_relu_block_1r/ReadVariableOp_1?9nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?;nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?(nodes_nn/bn1_relu_block_2/ReadVariableOp?*nodes_nn/bn1_relu_block_2/ReadVariableOp_1?&nodes_nn/conv2d/BiasAdd/ReadVariableOp?%nodes_nn/conv2d/Conv2D/ReadVariableOp?5nodes_nn/conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?4nodes_nn/conv2d_0_relu_block_1/Conv2D/ReadVariableOp?6nodes_nn/conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?5nodes_nn/conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?5nodes_nn/conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?4nodes_nn/conv2d_0_relu_block_2/Conv2D/ReadVariableOp?(nodes_nn/conv2d_1/BiasAdd/ReadVariableOp?'nodes_nn/conv2d_1/Conv2D/ReadVariableOp?5nodes_nn/conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?4nodes_nn/conv2d_1_relu_block_1/Conv2D/ReadVariableOp?6nodes_nn/conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?5nodes_nn/conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?5nodes_nn/conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?4nodes_nn/conv2d_1_relu_block_2/Conv2D/ReadVariableOp?(nodes_nn/conv2d_2/BiasAdd/ReadVariableOp?'nodes_nn/conv2d_2/Conv2D/ReadVariableOp?0nodes_nn/conv2d_transpose/BiasAdd/ReadVariableOp?9nodes_nn/conv2d_transpose/conv2d_transpose/ReadVariableOp?
4nodes_nn/conv2d_0_relu_block_1/Conv2D/ReadVariableOpReadVariableOp=nodes_nn_conv2d_0_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype026
4nodes_nn/conv2d_0_relu_block_1/Conv2D/ReadVariableOp?
%nodes_nn/conv2d_0_relu_block_1/Conv2DConv2Dinput<nodes_nn/conv2d_0_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2'
%nodes_nn/conv2d_0_relu_block_1/Conv2D?
5nodes_nn/conv2d_0_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp>nodes_nn_conv2d_0_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5nodes_nn/conv2d_0_relu_block_1/BiasAdd/ReadVariableOp?
&nodes_nn/conv2d_0_relu_block_1/BiasAddBiasAdd.nodes_nn/conv2d_0_relu_block_1/Conv2D:output:0=nodes_nn/conv2d_0_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2(
&nodes_nn/conv2d_0_relu_block_1/BiasAdd?
(nodes_nn/bn0_relu_block_1/ReadVariableOpReadVariableOp1nodes_nn_bn0_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(nodes_nn/bn0_relu_block_1/ReadVariableOp?
*nodes_nn/bn0_relu_block_1/ReadVariableOp_1ReadVariableOp3nodes_nn_bn0_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*nodes_nn/bn0_relu_block_1/ReadVariableOp_1?
9nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBnodes_nn_bn0_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02;
9nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp?
;nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDnodes_nn_bn0_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
*nodes_nn/bn0_relu_block_1/FusedBatchNormV3FusedBatchNormV3/nodes_nn/conv2d_0_relu_block_1/BiasAdd:output:00nodes_nn/bn0_relu_block_1/ReadVariableOp:value:02nodes_nn/bn0_relu_block_1/ReadVariableOp_1:value:0Anodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0Cnodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2,
*nodes_nn/bn0_relu_block_1/FusedBatchNormV3?
 nodes_nn/relu0_relu_block_1/ReluRelu.nodes_nn/bn0_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2"
 nodes_nn/relu0_relu_block_1/Relu?
4nodes_nn/conv2d_1_relu_block_1/Conv2D/ReadVariableOpReadVariableOp=nodes_nn_conv2d_1_relu_block_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype026
4nodes_nn/conv2d_1_relu_block_1/Conv2D/ReadVariableOp?
%nodes_nn/conv2d_1_relu_block_1/Conv2DConv2D.nodes_nn/relu0_relu_block_1/Relu:activations:0<nodes_nn/conv2d_1_relu_block_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2'
%nodes_nn/conv2d_1_relu_block_1/Conv2D?
5nodes_nn/conv2d_1_relu_block_1/BiasAdd/ReadVariableOpReadVariableOp>nodes_nn_conv2d_1_relu_block_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5nodes_nn/conv2d_1_relu_block_1/BiasAdd/ReadVariableOp?
&nodes_nn/conv2d_1_relu_block_1/BiasAddBiasAdd.nodes_nn/conv2d_1_relu_block_1/Conv2D:output:0=nodes_nn/conv2d_1_relu_block_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2(
&nodes_nn/conv2d_1_relu_block_1/BiasAdd?
(nodes_nn/bn1_relu_block_1/ReadVariableOpReadVariableOp1nodes_nn_bn1_relu_block_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(nodes_nn/bn1_relu_block_1/ReadVariableOp?
*nodes_nn/bn1_relu_block_1/ReadVariableOp_1ReadVariableOp3nodes_nn_bn1_relu_block_1_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*nodes_nn/bn1_relu_block_1/ReadVariableOp_1?
9nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOpReadVariableOpBnodes_nn_bn1_relu_block_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02;
9nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp?
;nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDnodes_nn_bn1_relu_block_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1?
*nodes_nn/bn1_relu_block_1/FusedBatchNormV3FusedBatchNormV3/nodes_nn/conv2d_1_relu_block_1/BiasAdd:output:00nodes_nn/bn1_relu_block_1/ReadVariableOp:value:02nodes_nn/bn1_relu_block_1/ReadVariableOp_1:value:0Anodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp:value:0Cnodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2,
*nodes_nn/bn1_relu_block_1/FusedBatchNormV3?
 nodes_nn/relu1_relu_block_1/ReluRelu.nodes_nn/bn1_relu_block_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2"
 nodes_nn/relu1_relu_block_1/Relu?
nodes_nn/max_pooling2d/MaxPoolMaxPool.nodes_nn/relu1_relu_block_1/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2 
nodes_nn/max_pooling2d/MaxPool?
nodes_nn/dropout/IdentityIdentity'nodes_nn/max_pooling2d/MaxPool:output:0*
T0*1
_output_shapes
:???????????2
nodes_nn/dropout/Identity?
4nodes_nn/conv2d_0_relu_block_2/Conv2D/ReadVariableOpReadVariableOp=nodes_nn_conv2d_0_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype026
4nodes_nn/conv2d_0_relu_block_2/Conv2D/ReadVariableOp?
%nodes_nn/conv2d_0_relu_block_2/Conv2DConv2D"nodes_nn/dropout/Identity:output:0<nodes_nn/conv2d_0_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2'
%nodes_nn/conv2d_0_relu_block_2/Conv2D?
5nodes_nn/conv2d_0_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp>nodes_nn_conv2d_0_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5nodes_nn/conv2d_0_relu_block_2/BiasAdd/ReadVariableOp?
&nodes_nn/conv2d_0_relu_block_2/BiasAddBiasAdd.nodes_nn/conv2d_0_relu_block_2/Conv2D:output:0=nodes_nn/conv2d_0_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2(
&nodes_nn/conv2d_0_relu_block_2/BiasAdd?
(nodes_nn/bn0_relu_block_2/ReadVariableOpReadVariableOp1nodes_nn_bn0_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02*
(nodes_nn/bn0_relu_block_2/ReadVariableOp?
*nodes_nn/bn0_relu_block_2/ReadVariableOp_1ReadVariableOp3nodes_nn_bn0_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*nodes_nn/bn0_relu_block_2/ReadVariableOp_1?
9nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBnodes_nn_bn0_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02;
9nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp?
;nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDnodes_nn_bn0_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
*nodes_nn/bn0_relu_block_2/FusedBatchNormV3FusedBatchNormV3/nodes_nn/conv2d_0_relu_block_2/BiasAdd:output:00nodes_nn/bn0_relu_block_2/ReadVariableOp:value:02nodes_nn/bn0_relu_block_2/ReadVariableOp_1:value:0Anodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0Cnodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2,
*nodes_nn/bn0_relu_block_2/FusedBatchNormV3?
 nodes_nn/relu0_relu_block_2/ReluRelu.nodes_nn/bn0_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2"
 nodes_nn/relu0_relu_block_2/Relu?
4nodes_nn/conv2d_1_relu_block_2/Conv2D/ReadVariableOpReadVariableOp=nodes_nn_conv2d_1_relu_block_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype026
4nodes_nn/conv2d_1_relu_block_2/Conv2D/ReadVariableOp?
%nodes_nn/conv2d_1_relu_block_2/Conv2DConv2D.nodes_nn/relu0_relu_block_2/Relu:activations:0<nodes_nn/conv2d_1_relu_block_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2'
%nodes_nn/conv2d_1_relu_block_2/Conv2D?
5nodes_nn/conv2d_1_relu_block_2/BiasAdd/ReadVariableOpReadVariableOp>nodes_nn_conv2d_1_relu_block_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5nodes_nn/conv2d_1_relu_block_2/BiasAdd/ReadVariableOp?
&nodes_nn/conv2d_1_relu_block_2/BiasAddBiasAdd.nodes_nn/conv2d_1_relu_block_2/Conv2D:output:0=nodes_nn/conv2d_1_relu_block_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2(
&nodes_nn/conv2d_1_relu_block_2/BiasAdd?
(nodes_nn/bn1_relu_block_2/ReadVariableOpReadVariableOp1nodes_nn_bn1_relu_block_2_readvariableop_resource*
_output_shapes
: *
dtype02*
(nodes_nn/bn1_relu_block_2/ReadVariableOp?
*nodes_nn/bn1_relu_block_2/ReadVariableOp_1ReadVariableOp3nodes_nn_bn1_relu_block_2_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*nodes_nn/bn1_relu_block_2/ReadVariableOp_1?
9nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOpReadVariableOpBnodes_nn_bn1_relu_block_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02;
9nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp?
;nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDnodes_nn_bn1_relu_block_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1?
*nodes_nn/bn1_relu_block_2/FusedBatchNormV3FusedBatchNormV3/nodes_nn/conv2d_1_relu_block_2/BiasAdd:output:00nodes_nn/bn1_relu_block_2/ReadVariableOp:value:02nodes_nn/bn1_relu_block_2/ReadVariableOp_1:value:0Anodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp:value:0Cnodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2,
*nodes_nn/bn1_relu_block_2/FusedBatchNormV3?
 nodes_nn/relu1_relu_block_2/ReluRelu.nodes_nn/bn1_relu_block_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2"
 nodes_nn/relu1_relu_block_2/Relu?
nodes_nn/conv2d_transpose/ShapeShape.nodes_nn/relu1_relu_block_2/Relu:activations:0*
T0*
_output_shapes
:2!
nodes_nn/conv2d_transpose/Shape?
-nodes_nn/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-nodes_nn/conv2d_transpose/strided_slice/stack?
/nodes_nn/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/nodes_nn/conv2d_transpose/strided_slice/stack_1?
/nodes_nn/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/nodes_nn/conv2d_transpose/strided_slice/stack_2?
'nodes_nn/conv2d_transpose/strided_sliceStridedSlice(nodes_nn/conv2d_transpose/Shape:output:06nodes_nn/conv2d_transpose/strided_slice/stack:output:08nodes_nn/conv2d_transpose/strided_slice/stack_1:output:08nodes_nn/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'nodes_nn/conv2d_transpose/strided_slice?
!nodes_nn/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2#
!nodes_nn/conv2d_transpose/stack/1?
!nodes_nn/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2#
!nodes_nn/conv2d_transpose/stack/2?
!nodes_nn/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!nodes_nn/conv2d_transpose/stack/3?
nodes_nn/conv2d_transpose/stackPack0nodes_nn/conv2d_transpose/strided_slice:output:0*nodes_nn/conv2d_transpose/stack/1:output:0*nodes_nn/conv2d_transpose/stack/2:output:0*nodes_nn/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2!
nodes_nn/conv2d_transpose/stack?
/nodes_nn/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/nodes_nn/conv2d_transpose/strided_slice_1/stack?
1nodes_nn/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1nodes_nn/conv2d_transpose/strided_slice_1/stack_1?
1nodes_nn/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1nodes_nn/conv2d_transpose/strided_slice_1/stack_2?
)nodes_nn/conv2d_transpose/strided_slice_1StridedSlice(nodes_nn/conv2d_transpose/stack:output:08nodes_nn/conv2d_transpose/strided_slice_1/stack:output:0:nodes_nn/conv2d_transpose/strided_slice_1/stack_1:output:0:nodes_nn/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)nodes_nn/conv2d_transpose/strided_slice_1?
9nodes_nn/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpBnodes_nn_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02;
9nodes_nn/conv2d_transpose/conv2d_transpose/ReadVariableOp?
*nodes_nn/conv2d_transpose/conv2d_transposeConv2DBackpropInput(nodes_nn/conv2d_transpose/stack:output:0Anodes_nn/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0.nodes_nn/relu1_relu_block_2/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2,
*nodes_nn/conv2d_transpose/conv2d_transpose?
0nodes_nn/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp9nodes_nn_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0nodes_nn/conv2d_transpose/BiasAdd/ReadVariableOp?
!nodes_nn/conv2d_transpose/BiasAddBiasAdd3nodes_nn/conv2d_transpose/conv2d_transpose:output:08nodes_nn/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2#
!nodes_nn/conv2d_transpose/BiasAdd?
 nodes_nn/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2"
 nodes_nn/concatenate/concat/axis?
nodes_nn/concatenate/concatConcatV2.nodes_nn/relu1_relu_block_1/Relu:activations:0*nodes_nn/conv2d_transpose/BiasAdd:output:0)nodes_nn/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? 2
nodes_nn/concatenate/concat?
5nodes_nn/conv2d_0_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp>nodes_nn_conv2d_0_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5nodes_nn/conv2d_0_relu_block_1r/Conv2D/ReadVariableOp?
&nodes_nn/conv2d_0_relu_block_1r/Conv2DConv2D$nodes_nn/concatenate/concat:output:0=nodes_nn/conv2d_0_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2(
&nodes_nn/conv2d_0_relu_block_1r/Conv2D?
6nodes_nn/conv2d_0_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp?nodes_nn_conv2d_0_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6nodes_nn/conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp?
'nodes_nn/conv2d_0_relu_block_1r/BiasAddBiasAdd/nodes_nn/conv2d_0_relu_block_1r/Conv2D:output:0>nodes_nn/conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2)
'nodes_nn/conv2d_0_relu_block_1r/BiasAdd?
)nodes_nn/bn0_relu_block_1r/ReadVariableOpReadVariableOp2nodes_nn_bn0_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02+
)nodes_nn/bn0_relu_block_1r/ReadVariableOp?
+nodes_nn/bn0_relu_block_1r/ReadVariableOp_1ReadVariableOp4nodes_nn_bn0_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02-
+nodes_nn/bn0_relu_block_1r/ReadVariableOp_1?
:nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOpCnodes_nn_bn0_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02<
:nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
<nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEnodes_nn_bn0_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02>
<nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
+nodes_nn/bn0_relu_block_1r/FusedBatchNormV3FusedBatchNormV30nodes_nn/conv2d_0_relu_block_1r/BiasAdd:output:01nodes_nn/bn0_relu_block_1r/ReadVariableOp:value:03nodes_nn/bn0_relu_block_1r/ReadVariableOp_1:value:0Bnodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0Dnodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2-
+nodes_nn/bn0_relu_block_1r/FusedBatchNormV3?
!nodes_nn/relu0_relu_block_1r/ReluRelu/nodes_nn/bn0_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2#
!nodes_nn/relu0_relu_block_1r/Relu?
5nodes_nn/conv2d_1_relu_block_1r/Conv2D/ReadVariableOpReadVariableOp>nodes_nn_conv2d_1_relu_block_1r_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5nodes_nn/conv2d_1_relu_block_1r/Conv2D/ReadVariableOp?
&nodes_nn/conv2d_1_relu_block_1r/Conv2DConv2D/nodes_nn/relu0_relu_block_1r/Relu:activations:0=nodes_nn/conv2d_1_relu_block_1r/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2(
&nodes_nn/conv2d_1_relu_block_1r/Conv2D?
6nodes_nn/conv2d_1_relu_block_1r/BiasAdd/ReadVariableOpReadVariableOp?nodes_nn_conv2d_1_relu_block_1r_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6nodes_nn/conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp?
'nodes_nn/conv2d_1_relu_block_1r/BiasAddBiasAdd/nodes_nn/conv2d_1_relu_block_1r/Conv2D:output:0>nodes_nn/conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2)
'nodes_nn/conv2d_1_relu_block_1r/BiasAdd?
)nodes_nn/bn1_relu_block_1r/ReadVariableOpReadVariableOp2nodes_nn_bn1_relu_block_1r_readvariableop_resource*
_output_shapes
:*
dtype02+
)nodes_nn/bn1_relu_block_1r/ReadVariableOp?
+nodes_nn/bn1_relu_block_1r/ReadVariableOp_1ReadVariableOp4nodes_nn_bn1_relu_block_1r_readvariableop_1_resource*
_output_shapes
:*
dtype02-
+nodes_nn/bn1_relu_block_1r/ReadVariableOp_1?
:nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOpReadVariableOpCnodes_nn_bn1_relu_block_1r_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02<
:nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp?
<nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEnodes_nn_bn1_relu_block_1r_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02>
<nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1?
+nodes_nn/bn1_relu_block_1r/FusedBatchNormV3FusedBatchNormV30nodes_nn/conv2d_1_relu_block_1r/BiasAdd:output:01nodes_nn/bn1_relu_block_1r/ReadVariableOp:value:03nodes_nn/bn1_relu_block_1r/ReadVariableOp_1:value:0Bnodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:value:0Dnodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2-
+nodes_nn/bn1_relu_block_1r/FusedBatchNormV3?
!nodes_nn/relu1_relu_block_1r/ReluRelu/nodes_nn/bn1_relu_block_1r/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2#
!nodes_nn/relu1_relu_block_1r/Relu?
'nodes_nn/conv2d_2/Conv2D/ReadVariableOpReadVariableOp0nodes_nn_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'nodes_nn/conv2d_2/Conv2D/ReadVariableOp?
nodes_nn/conv2d_2/Conv2DConv2D/nodes_nn/relu1_relu_block_1r/Relu:activations:0/nodes_nn/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
nodes_nn/conv2d_2/Conv2D?
(nodes_nn/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp1nodes_nn_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(nodes_nn/conv2d_2/BiasAdd/ReadVariableOp?
nodes_nn/conv2d_2/BiasAddBiasAdd!nodes_nn/conv2d_2/Conv2D:output:00nodes_nn/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
nodes_nn/conv2d_2/BiasAdd?
'nodes_nn/conv2d_1/Conv2D/ReadVariableOpReadVariableOp0nodes_nn_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'nodes_nn/conv2d_1/Conv2D/ReadVariableOp?
nodes_nn/conv2d_1/Conv2DConv2D/nodes_nn/relu1_relu_block_1r/Relu:activations:0/nodes_nn/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
nodes_nn/conv2d_1/Conv2D?
(nodes_nn/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp1nodes_nn_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(nodes_nn/conv2d_1/BiasAdd/ReadVariableOp?
nodes_nn/conv2d_1/BiasAddBiasAdd!nodes_nn/conv2d_1/Conv2D:output:00nodes_nn/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
nodes_nn/conv2d_1/BiasAdd?
%nodes_nn/conv2d/Conv2D/ReadVariableOpReadVariableOp.nodes_nn_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%nodes_nn/conv2d/Conv2D/ReadVariableOp?
nodes_nn/conv2d/Conv2DConv2D/nodes_nn/relu1_relu_block_1r/Relu:activations:0-nodes_nn/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
nodes_nn/conv2d/Conv2D?
&nodes_nn/conv2d/BiasAdd/ReadVariableOpReadVariableOp/nodes_nn_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&nodes_nn/conv2d/BiasAdd/ReadVariableOp?
nodes_nn/conv2d/BiasAddBiasAddnodes_nn/conv2d/Conv2D:output:0.nodes_nn/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
nodes_nn/conv2d/BiasAdd?
nodes_nn/node_types/SoftmaxSoftmax"nodes_nn/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
nodes_nn/node_types/Softmax?
nodes_nn/degrees/SoftmaxSoftmax"nodes_nn/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
nodes_nn/degrees/Softmax?
nodes_nn/node_pos/SigmoidSigmoid nodes_nn/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
nodes_nn/node_pos/Sigmoid?
IdentityIdentity"nodes_nn/degrees/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identitynodes_nn/node_pos/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identity%nodes_nn/node_types/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2?
NoOpNoOp:^nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp<^nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1)^nodes_nn/bn0_relu_block_1/ReadVariableOp+^nodes_nn/bn0_relu_block_1/ReadVariableOp_1;^nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp=^nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*^nodes_nn/bn0_relu_block_1r/ReadVariableOp,^nodes_nn/bn0_relu_block_1r/ReadVariableOp_1:^nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp<^nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1)^nodes_nn/bn0_relu_block_2/ReadVariableOp+^nodes_nn/bn0_relu_block_2/ReadVariableOp_1:^nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp<^nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1)^nodes_nn/bn1_relu_block_1/ReadVariableOp+^nodes_nn/bn1_relu_block_1/ReadVariableOp_1;^nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp=^nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1*^nodes_nn/bn1_relu_block_1r/ReadVariableOp,^nodes_nn/bn1_relu_block_1r/ReadVariableOp_1:^nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp<^nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1)^nodes_nn/bn1_relu_block_2/ReadVariableOp+^nodes_nn/bn1_relu_block_2/ReadVariableOp_1'^nodes_nn/conv2d/BiasAdd/ReadVariableOp&^nodes_nn/conv2d/Conv2D/ReadVariableOp6^nodes_nn/conv2d_0_relu_block_1/BiasAdd/ReadVariableOp5^nodes_nn/conv2d_0_relu_block_1/Conv2D/ReadVariableOp7^nodes_nn/conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp6^nodes_nn/conv2d_0_relu_block_1r/Conv2D/ReadVariableOp6^nodes_nn/conv2d_0_relu_block_2/BiasAdd/ReadVariableOp5^nodes_nn/conv2d_0_relu_block_2/Conv2D/ReadVariableOp)^nodes_nn/conv2d_1/BiasAdd/ReadVariableOp(^nodes_nn/conv2d_1/Conv2D/ReadVariableOp6^nodes_nn/conv2d_1_relu_block_1/BiasAdd/ReadVariableOp5^nodes_nn/conv2d_1_relu_block_1/Conv2D/ReadVariableOp7^nodes_nn/conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp6^nodes_nn/conv2d_1_relu_block_1r/Conv2D/ReadVariableOp6^nodes_nn/conv2d_1_relu_block_2/BiasAdd/ReadVariableOp5^nodes_nn/conv2d_1_relu_block_2/Conv2D/ReadVariableOp)^nodes_nn/conv2d_2/BiasAdd/ReadVariableOp(^nodes_nn/conv2d_2/Conv2D/ReadVariableOp1^nodes_nn/conv2d_transpose/BiasAdd/ReadVariableOp:^nodes_nn/conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp9nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp2z
;nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_1;nodes_nn/bn0_relu_block_1/FusedBatchNormV3/ReadVariableOp_12T
(nodes_nn/bn0_relu_block_1/ReadVariableOp(nodes_nn/bn0_relu_block_1/ReadVariableOp2X
*nodes_nn/bn0_relu_block_1/ReadVariableOp_1*nodes_nn/bn0_relu_block_1/ReadVariableOp_12x
:nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp:nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp2|
<nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1<nodes_nn/bn0_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12V
)nodes_nn/bn0_relu_block_1r/ReadVariableOp)nodes_nn/bn0_relu_block_1r/ReadVariableOp2Z
+nodes_nn/bn0_relu_block_1r/ReadVariableOp_1+nodes_nn/bn0_relu_block_1r/ReadVariableOp_12v
9nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp9nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp2z
;nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_1;nodes_nn/bn0_relu_block_2/FusedBatchNormV3/ReadVariableOp_12T
(nodes_nn/bn0_relu_block_2/ReadVariableOp(nodes_nn/bn0_relu_block_2/ReadVariableOp2X
*nodes_nn/bn0_relu_block_2/ReadVariableOp_1*nodes_nn/bn0_relu_block_2/ReadVariableOp_12v
9nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp9nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp2z
;nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_1;nodes_nn/bn1_relu_block_1/FusedBatchNormV3/ReadVariableOp_12T
(nodes_nn/bn1_relu_block_1/ReadVariableOp(nodes_nn/bn1_relu_block_1/ReadVariableOp2X
*nodes_nn/bn1_relu_block_1/ReadVariableOp_1*nodes_nn/bn1_relu_block_1/ReadVariableOp_12x
:nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp:nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp2|
<nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_1<nodes_nn/bn1_relu_block_1r/FusedBatchNormV3/ReadVariableOp_12V
)nodes_nn/bn1_relu_block_1r/ReadVariableOp)nodes_nn/bn1_relu_block_1r/ReadVariableOp2Z
+nodes_nn/bn1_relu_block_1r/ReadVariableOp_1+nodes_nn/bn1_relu_block_1r/ReadVariableOp_12v
9nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp9nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp2z
;nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_1;nodes_nn/bn1_relu_block_2/FusedBatchNormV3/ReadVariableOp_12T
(nodes_nn/bn1_relu_block_2/ReadVariableOp(nodes_nn/bn1_relu_block_2/ReadVariableOp2X
*nodes_nn/bn1_relu_block_2/ReadVariableOp_1*nodes_nn/bn1_relu_block_2/ReadVariableOp_12P
&nodes_nn/conv2d/BiasAdd/ReadVariableOp&nodes_nn/conv2d/BiasAdd/ReadVariableOp2N
%nodes_nn/conv2d/Conv2D/ReadVariableOp%nodes_nn/conv2d/Conv2D/ReadVariableOp2n
5nodes_nn/conv2d_0_relu_block_1/BiasAdd/ReadVariableOp5nodes_nn/conv2d_0_relu_block_1/BiasAdd/ReadVariableOp2l
4nodes_nn/conv2d_0_relu_block_1/Conv2D/ReadVariableOp4nodes_nn/conv2d_0_relu_block_1/Conv2D/ReadVariableOp2p
6nodes_nn/conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp6nodes_nn/conv2d_0_relu_block_1r/BiasAdd/ReadVariableOp2n
5nodes_nn/conv2d_0_relu_block_1r/Conv2D/ReadVariableOp5nodes_nn/conv2d_0_relu_block_1r/Conv2D/ReadVariableOp2n
5nodes_nn/conv2d_0_relu_block_2/BiasAdd/ReadVariableOp5nodes_nn/conv2d_0_relu_block_2/BiasAdd/ReadVariableOp2l
4nodes_nn/conv2d_0_relu_block_2/Conv2D/ReadVariableOp4nodes_nn/conv2d_0_relu_block_2/Conv2D/ReadVariableOp2T
(nodes_nn/conv2d_1/BiasAdd/ReadVariableOp(nodes_nn/conv2d_1/BiasAdd/ReadVariableOp2R
'nodes_nn/conv2d_1/Conv2D/ReadVariableOp'nodes_nn/conv2d_1/Conv2D/ReadVariableOp2n
5nodes_nn/conv2d_1_relu_block_1/BiasAdd/ReadVariableOp5nodes_nn/conv2d_1_relu_block_1/BiasAdd/ReadVariableOp2l
4nodes_nn/conv2d_1_relu_block_1/Conv2D/ReadVariableOp4nodes_nn/conv2d_1_relu_block_1/Conv2D/ReadVariableOp2p
6nodes_nn/conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp6nodes_nn/conv2d_1_relu_block_1r/BiasAdd/ReadVariableOp2n
5nodes_nn/conv2d_1_relu_block_1r/Conv2D/ReadVariableOp5nodes_nn/conv2d_1_relu_block_1r/Conv2D/ReadVariableOp2n
5nodes_nn/conv2d_1_relu_block_2/BiasAdd/ReadVariableOp5nodes_nn/conv2d_1_relu_block_2/BiasAdd/ReadVariableOp2l
4nodes_nn/conv2d_1_relu_block_2/Conv2D/ReadVariableOp4nodes_nn/conv2d_1_relu_block_2/Conv2D/ReadVariableOp2T
(nodes_nn/conv2d_2/BiasAdd/ReadVariableOp(nodes_nn/conv2d_2/BiasAdd/ReadVariableOp2R
'nodes_nn/conv2d_2/Conv2D/ReadVariableOp'nodes_nn/conv2d_2/Conv2D/ReadVariableOp2d
0nodes_nn/conv2d_transpose/BiasAdd/ReadVariableOp0nodes_nn/conv2d_transpose/BiasAdd/ReadVariableOp2v
9nodes_nn/conv2d_transpose/conv2d_transpose/ReadVariableOp9nodes_nn/conv2d_transpose/conv2d_transpose/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6657

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_degrees_layer_call_and_return_conditional_losses_6850

inputs
identitya
SoftmaxSoftmaxinputs*
T0*1
_output_shapes
:???????????2	
Softmaxo
IdentityIdentitySoftmax:softmax:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_6785

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_conv2d_0_relu_block_2_layer_call_and_return_conditional_losses_5963

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
B
&__inference_degrees_layer_call_fn_6855

inputs
identitya
SoftmaxSoftmaxinputs*
T0*1
_output_shapes
:???????????2	
Softmaxo
IdentityIdentitySoftmax:softmax:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_5936

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_conv2d_transpose_layer_call_fn_6413

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
/__inference_bn0_relu_block_1_layer_call_fn_5679

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
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
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_relu0_relu_block_1_layer_call_and_return_conditional_losses_5720

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5909

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?

"__inference_signature_wrapper_4833	
input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:$

unknown_25: 

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:$

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:$

unknown_39:

unknown_40:$

unknown_41:

unknown_42:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_42*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:???????????:???????????:???????????*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_11872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*1
_output_shapes
:???????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesw
u:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
4__inference_conv2d_0_relu_block_2_layer_call_fn_5973

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6027

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6465

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_relu0_relu_block_2_layer_call_and_return_conditional_losses_6122

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs"?.
saver_filename:0
Identity:0Identity_1018"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input8
serving_default_input:0???????????E
degrees:
StatefulPartitionedCall:0???????????F
node_pos:
StatefulPartitionedCall:1???????????H

node_types:
StatefulPartitionedCall:2???????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer_with_weights-12
layer-21
layer-22
layer_with_weights-13
layer-23
layer_with_weights-14
layer-24
layer_with_weights-15
layer-25
layer-26
layer-27
layer-28
	skips
	optimizer
 loss
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%
signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
r	variables
strainable_variables
tregularization_losses
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

zkernel
{bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
"
	optimizer
 "
trackable_dict_wrapper
?
&0
'1
-2
.3
/4
05
96
:7
@8
A9
B10
C11
T12
U13
[14
\15
]16
^17
g18
h19
n20
o21
p22
q23
z24
{25
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
?43"
trackable_list_wrapper
?
&0
'1
-2
.3
94
:5
@6
A7
T8
U9
[10
\11
g12
h13
n14
o15
z16
{17
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
?31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!	variables
"trainable_variables
?layers
?metrics
#regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
6:42conv2d_0_relu_block_1/kernel
(:&2conv2d_0_relu_block_1/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(	variables
)trainable_variables
?layers
?metrics
*regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
$:"2bn0_relu_block_1/gamma
#:!2bn0_relu_block_1/beta
,:* (2bn0_relu_block_1/moving_mean
0:. (2 bn0_relu_block_1/moving_variance
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1	variables
2trainable_variables
?layers
?metrics
3regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5	variables
6trainable_variables
?layers
?metrics
7regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:42conv2d_1_relu_block_1/kernel
(:&2conv2d_1_relu_block_1/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;	variables
<trainable_variables
?layers
?metrics
=regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
$:"2bn1_relu_block_1/gamma
#:!2bn1_relu_block_1/beta
,:* (2bn1_relu_block_1/moving_mean
0:. (2 bn1_relu_block_1/moving_variance
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
D	variables
Etrainable_variables
?layers
?metrics
Fregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
H	variables
Itrainable_variables
?layers
?metrics
Jregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
L	variables
Mtrainable_variables
?layers
?metrics
Nregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
P	variables
Qtrainable_variables
?layers
?metrics
Rregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:4 2conv2d_0_relu_block_2/kernel
(:& 2conv2d_0_relu_block_2/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
V	variables
Wtrainable_variables
?layers
?metrics
Xregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
$:" 2bn0_relu_block_2/gamma
#:! 2bn0_relu_block_2/beta
,:*  (2bn0_relu_block_2/moving_mean
0:.  (2 bn0_relu_block_2/moving_variance
<
[0
\1
]2
^3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_	variables
`trainable_variables
?layers
?metrics
aregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
c	variables
dtrainable_variables
?layers
?metrics
eregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:4  2conv2d_1_relu_block_2/kernel
(:& 2conv2d_1_relu_block_2/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
i	variables
jtrainable_variables
?layers
?metrics
kregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
$:" 2bn1_relu_block_2/gamma
#:! 2bn1_relu_block_2/beta
,:*  (2bn1_relu_block_2/moving_mean
0:.  (2 bn1_relu_block_2/moving_variance
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
r	variables
strainable_variables
?layers
?metrics
tregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
v	variables
wtrainable_variables
?layers
?metrics
xregularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/ 2conv2d_transpose/kernel
#:!2conv2d_transpose/bias
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
|	variables
}trainable_variables
?layers
?metrics
~regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5 2conv2d_0_relu_block_1r/kernel
):'2conv2d_0_relu_block_1r/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2bn0_relu_block_1r/gamma
$:"2bn0_relu_block_1r/beta
-:+ (2bn0_relu_block_1r/moving_mean
1:/ (2!bn0_relu_block_1r/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:52conv2d_1_relu_block_1r/kernel
):'2conv2d_1_relu_block_1r/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2bn1_relu_block_1r/gamma
$:"2bn1_relu_block_1r/beta
-:+ (2bn1_relu_block_1r/moving_mean
1:/ (2!bn1_relu_block_1r/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
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
?	variables
?trainable_variables
?layers
?metrics
?regularization_losses
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
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
28"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
z
/0
01
B2
C3
]4
^5
p6
q7
?8
?9
?10
?11"
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
/0
01"
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
B0
C1"
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
]0
^1"
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
p0
q1"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
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
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
B__inference_nodes_nn_layer_call_and_return_conditional_losses_5009
B__inference_nodes_nn_layer_call_and_return_conditional_losses_5192
B__inference_nodes_nn_layer_call_and_return_conditional_losses_4320
B__inference_nodes_nn_layer_call_and_return_conditional_losses_4503?
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
?2?
'__inference_nodes_nn_layer_call_fn_2396
'__inference_nodes_nn_layer_call_fn_5368
'__inference_nodes_nn_layer_call_fn_5551
'__inference_nodes_nn_layer_call_fn_4144?
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
__inference__wrapped_model_1187input"?
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
?2?
O__inference_conv2d_0_relu_block_1_layer_call_and_return_conditional_losses_5561?
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
4__inference_conv2d_0_relu_block_1_layer_call_fn_5571?
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
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5589
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5607
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5625
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5643?
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
/__inference_bn0_relu_block_1_layer_call_fn_5661
/__inference_bn0_relu_block_1_layer_call_fn_5679
/__inference_bn0_relu_block_1_layer_call_fn_5697
/__inference_bn0_relu_block_1_layer_call_fn_5715?
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
L__inference_relu0_relu_block_1_layer_call_and_return_conditional_losses_5720?
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
1__inference_relu0_relu_block_1_layer_call_fn_5725?
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
O__inference_conv2d_1_relu_block_1_layer_call_and_return_conditional_losses_5735?
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
4__inference_conv2d_1_relu_block_1_layer_call_fn_5745?
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
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5763
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5781
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5799
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5817?
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
/__inference_bn1_relu_block_1_layer_call_fn_5835
/__inference_bn1_relu_block_1_layer_call_fn_5853
/__inference_bn1_relu_block_1_layer_call_fn_5871
/__inference_bn1_relu_block_1_layer_call_fn_5889?
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
L__inference_relu1_relu_block_1_layer_call_and_return_conditional_losses_5894?
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
1__inference_relu1_relu_block_1_layer_call_fn_5899?
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
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5904
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5909?
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
,__inference_max_pooling2d_layer_call_fn_5914
,__inference_max_pooling2d_layer_call_fn_5919?
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
A__inference_dropout_layer_call_and_return_conditional_losses_5924
A__inference_dropout_layer_call_and_return_conditional_losses_5936?
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
&__inference_dropout_layer_call_fn_5941
&__inference_dropout_layer_call_fn_5953?
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
O__inference_conv2d_0_relu_block_2_layer_call_and_return_conditional_losses_5963?
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
4__inference_conv2d_0_relu_block_2_layer_call_fn_5973?
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
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_5991
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6009
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6027
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6045?
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
/__inference_bn0_relu_block_2_layer_call_fn_6063
/__inference_bn0_relu_block_2_layer_call_fn_6081
/__inference_bn0_relu_block_2_layer_call_fn_6099
/__inference_bn0_relu_block_2_layer_call_fn_6117?
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
L__inference_relu0_relu_block_2_layer_call_and_return_conditional_losses_6122?
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
1__inference_relu0_relu_block_2_layer_call_fn_6127?
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
O__inference_conv2d_1_relu_block_2_layer_call_and_return_conditional_losses_6137?
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
4__inference_conv2d_1_relu_block_2_layer_call_fn_6147?
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
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6165
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6183
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6201
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6219?
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
/__inference_bn1_relu_block_2_layer_call_fn_6237
/__inference_bn1_relu_block_2_layer_call_fn_6255
/__inference_bn1_relu_block_2_layer_call_fn_6273
/__inference_bn1_relu_block_2_layer_call_fn_6291?
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
L__inference_relu1_relu_block_2_layer_call_and_return_conditional_losses_6296?
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
1__inference_relu1_relu_block_2_layer_call_fn_6301?
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
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6334
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6357?
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
/__inference_conv2d_transpose_layer_call_fn_6390
/__inference_conv2d_transpose_layer_call_fn_6413?
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
E__inference_concatenate_layer_call_and_return_conditional_losses_6420?
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
*__inference_concatenate_layer_call_fn_6427?
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
P__inference_conv2d_0_relu_block_1r_layer_call_and_return_conditional_losses_6437?
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
5__inference_conv2d_0_relu_block_1r_layer_call_fn_6447?
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
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6465
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6483
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6501
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6519?
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
?2?
0__inference_bn0_relu_block_1r_layer_call_fn_6537
0__inference_bn0_relu_block_1r_layer_call_fn_6555
0__inference_bn0_relu_block_1r_layer_call_fn_6573
0__inference_bn0_relu_block_1r_layer_call_fn_6591?
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
M__inference_relu0_relu_block_1r_layer_call_and_return_conditional_losses_6596?
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
2__inference_relu0_relu_block_1r_layer_call_fn_6601?
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
P__inference_conv2d_1_relu_block_1r_layer_call_and_return_conditional_losses_6611?
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
5__inference_conv2d_1_relu_block_1r_layer_call_fn_6621?
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
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6639
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6657
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6675
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6693?
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
?2?
0__inference_bn1_relu_block_1r_layer_call_fn_6711
0__inference_bn1_relu_block_1r_layer_call_fn_6729
0__inference_bn1_relu_block_1r_layer_call_fn_6747
0__inference_bn1_relu_block_1r_layer_call_fn_6765?
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
M__inference_relu1_relu_block_1r_layer_call_and_return_conditional_losses_6770?
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
2__inference_relu1_relu_block_1r_layer_call_fn_6775?
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
@__inference_conv2d_layer_call_and_return_conditional_losses_6785?
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
%__inference_conv2d_layer_call_fn_6795?
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
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6805?
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
'__inference_conv2d_1_layer_call_fn_6815?
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
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6825?
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
'__inference_conv2d_2_layer_call_fn_6835?
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
B__inference_node_pos_layer_call_and_return_conditional_losses_6840?
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
'__inference_node_pos_layer_call_fn_6845?
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
A__inference_degrees_layer_call_and_return_conditional_losses_6850?
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
&__inference_degrees_layer_call_fn_6855?
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
D__inference_node_types_layer_call_and_return_conditional_losses_6860?
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
)__inference_node_types_layer_call_fn_6865?
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
?B?
"__inference_signature_wrapper_4833input"?
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
 ?
__inference__wrapped_model_1187?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????8?5
.?+
)?&
input???????????
? "???
6
degrees+?(
degrees???????????
8
node_pos,?)
node_pos???????????
<

node_types.?+

node_types????????????
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5589?-./0M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5607?-./0M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5625v-./0=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
J__inference_bn0_relu_block_1_layer_call_and_return_conditional_losses_5643v-./0=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
/__inference_bn0_relu_block_1_layer_call_fn_5661?-./0M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
/__inference_bn0_relu_block_1_layer_call_fn_5679?-./0M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
/__inference_bn0_relu_block_1_layer_call_fn_5697i-./0=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
/__inference_bn0_relu_block_1_layer_call_fn_5715i-./0=?:
3?0
*?'
inputs???????????
p
? ""?????????????
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6465?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6483?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6501z????=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
K__inference_bn0_relu_block_1r_layer_call_and_return_conditional_losses_6519z????=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
0__inference_bn0_relu_block_1r_layer_call_fn_6537?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
0__inference_bn0_relu_block_1r_layer_call_fn_6555?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
0__inference_bn0_relu_block_1r_layer_call_fn_6573m????=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
0__inference_bn0_relu_block_1r_layer_call_fn_6591m????=?:
3?0
*?'
inputs???????????
p
? ""?????????????
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_5991?[\]^M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6009?[\]^M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6027v[\]^=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
J__inference_bn0_relu_block_2_layer_call_and_return_conditional_losses_6045v[\]^=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
/__inference_bn0_relu_block_2_layer_call_fn_6063?[\]^M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
/__inference_bn0_relu_block_2_layer_call_fn_6081?[\]^M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
/__inference_bn0_relu_block_2_layer_call_fn_6099i[\]^=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
/__inference_bn0_relu_block_2_layer_call_fn_6117i[\]^=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5763?@ABCM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5781?@ABCM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5799v@ABC=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
J__inference_bn1_relu_block_1_layer_call_and_return_conditional_losses_5817v@ABC=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
/__inference_bn1_relu_block_1_layer_call_fn_5835?@ABCM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
/__inference_bn1_relu_block_1_layer_call_fn_5853?@ABCM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
/__inference_bn1_relu_block_1_layer_call_fn_5871i@ABC=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
/__inference_bn1_relu_block_1_layer_call_fn_5889i@ABC=?:
3?0
*?'
inputs???????????
p
? ""?????????????
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6639?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6657?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6675z????=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
K__inference_bn1_relu_block_1r_layer_call_and_return_conditional_losses_6693z????=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
0__inference_bn1_relu_block_1r_layer_call_fn_6711?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
0__inference_bn1_relu_block_1r_layer_call_fn_6729?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
0__inference_bn1_relu_block_1r_layer_call_fn_6747m????=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
0__inference_bn1_relu_block_1r_layer_call_fn_6765m????=?:
3?0
*?'
inputs???????????
p
? ""?????????????
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6165?nopqM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6183?nopqM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6201vnopq=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
J__inference_bn1_relu_block_2_layer_call_and_return_conditional_losses_6219vnopq=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
/__inference_bn1_relu_block_2_layer_call_fn_6237?nopqM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
/__inference_bn1_relu_block_2_layer_call_fn_6255?nopqM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
/__inference_bn1_relu_block_2_layer_call_fn_6273inopq=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
/__inference_bn1_relu_block_2_layer_call_fn_6291inopq=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
E__inference_concatenate_layer_call_and_return_conditional_losses_6420?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? "/?,
%?"
0??????????? 
? ?
*__inference_concatenate_layer_call_fn_6427?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""???????????? ?
O__inference_conv2d_0_relu_block_1_layer_call_and_return_conditional_losses_5561p&'9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
4__inference_conv2d_0_relu_block_1_layer_call_fn_5571c&'9?6
/?,
*?'
inputs???????????
? ""?????????????
P__inference_conv2d_0_relu_block_1r_layer_call_and_return_conditional_losses_6437r??9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????
? ?
5__inference_conv2d_0_relu_block_1r_layer_call_fn_6447e??9?6
/?,
*?'
inputs??????????? 
? ""?????????????
O__inference_conv2d_0_relu_block_2_layer_call_and_return_conditional_losses_5963pTU9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
4__inference_conv2d_0_relu_block_2_layer_call_fn_5973cTU9?6
/?,
*?'
inputs???????????
? ""???????????? ?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6805r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
'__inference_conv2d_1_layer_call_fn_6815e??9?6
/?,
*?'
inputs???????????
? ""?????????????
O__inference_conv2d_1_relu_block_1_layer_call_and_return_conditional_losses_5735p9:9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
4__inference_conv2d_1_relu_block_1_layer_call_fn_5745c9:9?6
/?,
*?'
inputs???????????
? ""?????????????
P__inference_conv2d_1_relu_block_1r_layer_call_and_return_conditional_losses_6611r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
5__inference_conv2d_1_relu_block_1r_layer_call_fn_6621e??9?6
/?,
*?'
inputs???????????
? ""?????????????
O__inference_conv2d_1_relu_block_2_layer_call_and_return_conditional_losses_6137pgh9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
4__inference_conv2d_1_relu_block_2_layer_call_fn_6147cgh9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6825r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
'__inference_conv2d_2_layer_call_fn_6835e??9?6
/?,
*?'
inputs???????????
? ""?????????????
@__inference_conv2d_layer_call_and_return_conditional_losses_6785r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_conv2d_layer_call_fn_6795e??9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6334?z{I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6357pz{9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????
? ?
/__inference_conv2d_transpose_layer_call_fn_6390?z{I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
/__inference_conv2d_transpose_layer_call_fn_6413cz{9?6
/?,
*?'
inputs??????????? 
? ""?????????????
A__inference_degrees_layer_call_and_return_conditional_losses_6850l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
&__inference_degrees_layer_call_fn_6855_9?6
/?,
*?'
inputs???????????
? ""?????????????
A__inference_dropout_layer_call_and_return_conditional_losses_5924p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_5936p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
&__inference_dropout_layer_call_fn_5941c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
&__inference_dropout_layer_call_fn_5953c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5904?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5909l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_max_pooling2d_layer_call_fn_5914?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
,__inference_max_pooling2d_layer_call_fn_5919_9?6
/?,
*?'
inputs???????????
? ""?????????????
B__inference_node_pos_layer_call_and_return_conditional_losses_6840l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
'__inference_node_pos_layer_call_fn_6845_9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_node_types_layer_call_and_return_conditional_losses_6860l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_node_types_layer_call_fn_6865_9?6
/?,
*?'
inputs???????????
? ""?????????????
B__inference_nodes_nn_layer_call_and_return_conditional_losses_4320?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????@?=
6?3
)?&
input???????????
p 

 
? "???
~?{
'?$
0/0???????????
'?$
0/1???????????
'?$
0/2???????????
? ?
B__inference_nodes_nn_layer_call_and_return_conditional_losses_4503?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????@?=
6?3
)?&
input???????????
p

 
? "???
~?{
'?$
0/0???????????
'?$
0/1???????????
'?$
0/2???????????
? ?
B__inference_nodes_nn_layer_call_and_return_conditional_losses_5009?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "???
~?{
'?$
0/0???????????
'?$
0/1???????????
'?$
0/2???????????
? ?
B__inference_nodes_nn_layer_call_and_return_conditional_losses_5192?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????A?>
7?4
*?'
inputs???????????
p

 
? "???
~?{
'?$
0/0???????????
'?$
0/1???????????
'?$
0/2???????????
? ?
'__inference_nodes_nn_layer_call_fn_2396?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????@?=
6?3
)?&
input???????????
p 

 
? "x?u
%?"
0???????????
%?"
1???????????
%?"
2????????????
'__inference_nodes_nn_layer_call_fn_4144?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????@?=
6?3
)?&
input???????????
p

 
? "x?u
%?"
0???????????
%?"
1???????????
%?"
2????????????
'__inference_nodes_nn_layer_call_fn_5368?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "x?u
%?"
0???????????
%?"
1???????????
%?"
2????????????
'__inference_nodes_nn_layer_call_fn_5551?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????A?>
7?4
*?'
inputs???????????
p

 
? "x?u
%?"
0???????????
%?"
1???????????
%?"
2????????????
L__inference_relu0_relu_block_1_layer_call_and_return_conditional_losses_5720l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
1__inference_relu0_relu_block_1_layer_call_fn_5725_9?6
/?,
*?'
inputs???????????
? ""?????????????
M__inference_relu0_relu_block_1r_layer_call_and_return_conditional_losses_6596l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
2__inference_relu0_relu_block_1r_layer_call_fn_6601_9?6
/?,
*?'
inputs???????????
? ""?????????????
L__inference_relu0_relu_block_2_layer_call_and_return_conditional_losses_6122l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
1__inference_relu0_relu_block_2_layer_call_fn_6127_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
L__inference_relu1_relu_block_1_layer_call_and_return_conditional_losses_5894l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
1__inference_relu1_relu_block_1_layer_call_fn_5899_9?6
/?,
*?'
inputs???????????
? ""?????????????
M__inference_relu1_relu_block_1r_layer_call_and_return_conditional_losses_6770l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
2__inference_relu1_relu_block_1r_layer_call_fn_6775_9?6
/?,
*?'
inputs???????????
? ""?????????????
L__inference_relu1_relu_block_2_layer_call_and_return_conditional_losses_6296l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
1__inference_relu1_relu_block_2_layer_call_fn_6301_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
"__inference_signature_wrapper_4833?>&'-./09:@ABCTU[\]^ghnopqz{??????????????????A?>
? 
7?4
2
input)?&
input???????????"???
6
degrees+?(
degrees???????????
8
node_pos,?)
node_pos???????????
<

node_types.?+

node_types???????????