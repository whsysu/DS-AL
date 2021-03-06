#### #位运算

```python
0.基本原理

0s 表示一串 0，1s 表示一串 1。
x ^ 0s = x      x & 0s = 0      x | 0s = x
x ^ 1s = ~x     x & 1s = x      x | 1s = 1s
x ^ x  = 0      x & x = x       x | x = x

口诀总结：总共四大位运算
x:异或加1s可翻转，异或自己去重复
x:与mask得mask中的1来掩码，与x-1除最后1；与-x得最后1
x:或mask得所有的1来设值
x:左移乘2右移除


1.异或运算技巧：翻转+去重复
a.利用 x ^ 1s = ~x 的特点，可以将一个数的位级表示翻转；
b.利用 x ^ x = 0 的特点，可以将三个数中重复的两个数去除，只留下另一个		1^1^2 = 2
    
2.与运算技巧：掩码+去除最低位1+得到最低位1
a.利用 x & 0s = 0 和 x & 1s = x 的特点，可以实现掩码操作。一个数 num 与 mask：00111100 进行位与操作，只保留 num 中与 mask 的 1 部分相对应的位。
        01011011 &
        00111100
        --------
        00011000
b.n&(n-1) 去除 n 的位级表示中最低的那一位 1。例如对于二进制表示 01011011，减去 1 得到 01011010，这两个数相与得到 01011010。
        01011011 &
        01011010
        --------
        01011010
c.n&(-n) 得到 n 的位级表示中最低的那一位 1。-n 得到 n 的反码加 1，也就是 -n=~n+1。例如对于二进制表示 10110100，-n 得到 01001100，相与得到 00000100。
        10110100 &
        01001100
        --------
        00000100
n-(n&(-n)) 则可以去除 n 的位级表示中最低的那一位 1，和 n&(n-1) 效果一样。   

3.或运算技巧：设值+
a.利用 x | 0s = x 和 x | 1s = 1s 的特点，可以实现设值操作。一个数 num 与 mask：00111100 进行位或操作，将 num 中与 mask 的 1 部分相对应的位都设置为 1。
        01011011 |
        00111100
        --------
        01111111


4.移位运算

a. >> n 为算术右移，相当于除以 2n，例如 -7 >> 2 = -2。
        11111111111111111111111111111001  >> 2
        --------
        11111111111111111111111111111110
b. >>> n 为无符号右移，左边会补上 0。例如 -7 >>> 2 = 1073741822。
        11111111111111111111111111111001  >>> 2
        --------
        00111111111111111111111111111111
c. << n 为算术左移，相当于乘以 2n。-7 << 2 = -28。
        11111111111111111111111111111001  << 2
        --------
        11111111111111111111111111100100
        
4.mask 计算
a.要获取 111111111，将 0 取反即可，~0。

b.要得到只有第 i 位为 1 的 mask，将 1 向左移动 i-1 位即可，1<<(i-1) 。
	例如 1<<4 得到只有第 5 位为 1 的 mask ：00010000。

c.要得到 1 到 i 位为 1 的 mask，(1<<i)-1 即可，例如将 (1<<4)-1 = 00010000-1 = 00001111。

d.要得到 1 到 i 位为 0 的 mask，只需将 1 到 i 位为 1 的 mask 取反，即 ~((1<<i)-1)。
```

```python

#461. 汉明距离:整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
def hammingDistance(self, x: int, y: int) -> int:
    '''二进制：首先回顾一遍位运算的四种操作，异或/与/或/位移'''
    z=x^y  #这样不同的位置变为1，相同的位置变为0
    #统计z中1有多少个就行=>与运算
    res=0
    while z:
        z=z&(z-1)
        res+=1
    return res

#136. 只出现一次的数字:非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
def singleNumber(self, nums: List[int]) -> int:
    '''数组问题：首先想到双指针+排序+查找
                本题思路：1.先排序再查找 O(NlogN)
                2.hashmap 统计次数，再遍历一次即可
                3.技巧法:位运算：利用异或运算的性质'''
    res=0
    for i in nums:
        res^=i
    return res

#268. 缺失数字:给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。
def missingNumber(self, nums: List[int]) -> int:
    '''数组问题：首先想到双指针+查找+排序
        技巧：位运算+下标法+大数据处理法：hash/bitmap等，要做好总结
        由于本题的特殊性:考虑使用数学方法，求和运算,但是超大数据求和不太合适'''
    #n=len(nums)
    #return (1+n)*n//2-sum(nums)
    #使用位运算，将数组翻倍变成仅有一个单位数，其余都是双位数
    res=0
    n=len(nums)
    for i in range(0,n):
        res=res^nums[i]^i
    return res^n


#260. 只出现一次的数字:给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。
import collections
def singleNumber(self, nums: List[int]) -> List[int]:
    '''数组问题:首先想到排序+查找+搜索
        再次技巧性：位运算+
        本题：hashmap/数组 统计次数,但是空间有待优化'''
    #hashmap=collections.Counter(nums)
    #return [i for i in hashmap if hashmap[i]==1]

    #位运算 
    #第一步：所有数异或，所有数异或之后的值就是两个只出现一次的数a,b异或后的值s。
    #第二步：那我们用s & (-s) 可以得出s最低位为1的bit位对应的数k，
    #         这里有一个很关键点就是：这个为1的bit位肯定只来之a.b其中的一个数字中的对应二进制位的1
    #第三步：得到s&(-s)之后在对所有数进行&操作的话，就意味着可以将a和b区分在0和1的两个组
    #第四步：经过第三步之后就变成了只有一个数字存在单个其他都存在双个的数组，然后分别对两个组的所有数字再进行异或即可
    s=0
    for i in nums:
        s^=i   #第一步
    k = s & (-s) # 第二步  k就是s最后一个1
    result = [0]*2
    for i in nums:
        if i&k==0:  #一个数 num 与 mask 与操作，只保留 num 中与 mask 的 1 部分相对应的位。
            result[0]^=i
        else:
            result[1]^=i
    return result


#6. 不用额外变量交换两个整数;python 中更加简单 a,b=b,a 
a = a ^ b
b = a ^ b
a = a ^ b


#判断它是否是 2 的幂次方。
def isPowerOfTwo(self, n: int) -> bool:
    '''位运算中位移对应着除法运算：'''
    if n==0:
        return False
    return n&(n-1)==0  #2的幂仅有一位是1

#判断它是否是 4 的幂次方。
from math import log2
def isPowerOfFour(self, num: int) -> bool:
    '''递归或者循环判断
        位运算判断：一次右移2'''
    return num > 0 and log2(num) % 2 == 0

#693. 交替位二进制数 :二进制数相邻的两个位数永不相等。
def hasAlternatingBits(self, n: int) -> bool:
    '''整数转为二进制：'''
    string=bin(n).replace('0b','')
    for i in range(0,len(string)-1):
        if string[i]==string[i+1]:
            return False
    return True



```

