# #数学系列

<img src="C:\Users\WangHuan\AppData\Roaming\Typora\typora-user-images\image-20200624161517333.png" alt="image-20200624161517333" style="zoom:50%;" />



```python
#204. 计数质数:统计所有小于非负整数 n 的质数的数量。
def countPrimes(self, n: int) -> int:
    '''质数:一个数如果只能被 1 和它本身整除，那么这个数就是素数。
        =>2,3,5,7,9,11,13...
        本题思路：找到质数生成法则即可'''
    if n<2:
        return 0
    isPrim=[1]*n
    isPrim[0]=isPrim[1]=0
    # 埃式筛，把不大于根号n的所有质数的倍数剔除
    for i in range(2,int(sqrt(n))+1):
        if isPrim[i]:
            for j in range(i*i,n,i):
                isPrim[j]=0

    return sum(isPrim)

#2. 最大公约数
def gcd(a, b):
    return a if b==0 else gcd(b,a%b)
#最小公倍数为两数的乘积除以最大公约数。
def lcm(a, b):
    return a * b / gcd(a, b)

#504. 七进制数:给定一个整数，将其转化为7进制，并以字符串形式输出。
def convertToBase7(self, num: int) -> str:
    base7 = []
    flag = 0
    if num < 0:
        flag = 1
        num = -num

    while num >= 7:
        fig = str(num % 7)
        base7.append(fig)
        num = num // 7
    base7.append(str(num))
    if flag:
        base7.append('-')
    base7.reverse()
    return ''.join(base7)

#给定一个整数 n，返回 n! 结果尾数中零的数量。
def trailingZeroes(self, n: int) -> int:
    '''思路：首先阶乘的结果非常大，不适合直接计算，故想新的方法=>
    2*5=10所以有多少对2和5，就有多少0,而且5的个数一定要少于2的个数，所以只要找到有多少个5就可以。'''
    count = 0
    while n > 0:
        count += n // 5
        n //= 5

    return count

#67. 二进制求和 :两个二进制字符串，返回它们的和（用二进制表示）。
def addBinary(self, a: str, b: str) -> str:
    '''之前有链表求和，数组求和，字符串求和，和这题本质一致'''
    m, n = len(a), len(b)
    size = max(m, n)
    dif = abs(m - n)
    # 保证二个等长度
    if m > n:
        b = '0' * dif + b
    else:
        a = '0' * dif + a

    up = 0  # 表示是否进位
    res = ''
    for i in range(size - 1, -1, -1):
        temp = up + int(a[i]) + int(b[i])
        if temp > 1:
            up = 1
            temp = 0 if temp == 2 else 1
        else:
            up = 0
        res = str(temp) + res
    if up == 1:
        res = '1' + res
    return res


#462. 最少移动次数使数组元素相等,非空整数数组，找到使所有数组元素相等所需的最小移动数，其中每次移动可将选定的一个元素加1或减1
def minMoves2(self, nums: List[int]) -> int:
    '''思路：数组问题首先想到双指针+查找+排序+二分
        最少=>首先想到动态规划
        对本题：最终的数字一定位于元素中间，且一定是中位数！=>why
        暴力解法：假设最终的数从最小到最大遍历=>找到每种移动次数'''

    '''这题不用想什么中位数：设 a <= x <= b，将 a 和 b 都变化成 x 为最终目的，则需要步数为 x-a+b-x = b-a，		即两个数最后相等的话步数一定是他们的差，x 在 a 和 b 间任意取；
		所以最后剩的其实就是中位数；那么直接排序后首尾指针计算就好：'''
    n=len(nums)
    res=0
    nums.sort()
    for i in range(n//2):
        res+=nums[n-i-1]-nums[i]
    return res
'''方法二：使用快速选择寻找中位数:快速选择算法借鉴了快速排序的思想'''
public int minMoves2(int[] nums) {
    int move = 0;
    int median = findKthSmallest(nums, nums.length / 2);
    for (int num : nums) {
        move += Math.abs(num - median);
    }
    return move;
}

private int findKthSmallest(int[] nums, int k) {
    int l = 0, h = nums.length - 1;
    while (l < h) {
        int j = partition(nums, l, h);
        if (j == k) {
            break;
        }
        if (j < k) {
            l = j + 1;
        } else {
            h = j - 1;
        }
    }
    return nums[k];
}

private int partition(int[] nums, int l, int h) {
    int i = l, j = h + 1;
    while (true) {
        while (nums[++i] < nums[l] && i < h) ;
        while (nums[--j] > nums[l] && j > l) ;
        if (i >= j) {
            break;
        }
        swap(nums, i, j);
    }
    swap(nums, l, j);
    return j;
}

private void swap(int[] nums, int i, int j) {
    int tmp = nums[i];
    nums[i] = nums[j];
    nums[j] = tmp;
}


#169. 多数元素:多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素
def majorityElement(self, nums: List[int]) -> int:
    # 思路：先排序 找到中间元素在判断 Nlog(N)
    # hash直接统计每个元素的个数 
    # 分治:如果数 a 是数组 nums 的众数，如果我们将 nums 分成两部分，那么 a 必定是至少一部分的众数。
    n = len(nums)
    freq = {}
    for i in nums:
        freq[i] = 1 + freq.get(i, 0)
    _max = 0
    res = 0
    for val, count in freq.items():
        if count > _max:
            res = val
            _max = count
    return res
def majorityElement(self, nums):
    counts = collections.Counter(nums)
    return max(counts.keys(), key=counts.get)
'''可以利用 Boyer-Moore Majority Vote Algorithm 来解决这个问题，使得时间复杂度为 O(N)。可以这么理解该算法：使用 cnt 来统计一个元素出现的次数，当遍历到的元素和统计元素不相等时，令 cnt--。如果前面查找了 i 个元素，且 cnt == 0，说明前 i 个元素没有 majority，或者有 majority，但是出现的次数少于 i / 2，因为如果多于 i / 2 的话 cnt 就一定不会为 0。此时剩下的 n - i 个元素中，majority 的数目依然多于 (n - i) / 2，因此继续查找就能找出 majority。'''
public int majorityElement(int[] nums) {
    int cnt = 0, majority = nums[0];
    for (int num : nums) {
        majority = (cnt == 0) ? num : majority;
        cnt = (majority == num) ? cnt + 1 : cnt - 1;
    }
    return majority;
}

#367. 有效的完全平方数:如果 num 是一个完全平方数，则返回 True，否则返回 False。
def isPerfectSquare(self, num: int) -> bool:
    '''思路1：最简单是遍历，从1开始，复杂度O(N):
   思路2：1,4,9,16,25=>相差3,5,7,9等差数列
   思路3:二分查找法'''
    left, right = 1, num // 2 + 1
    while left <= right:
        m = (left + right) // 2
        if m * m == num:
            return True
        elif m * m < num:
            left = m + 1
        else:
            right = m - 1

    return False

    # i = 1
    # while num > 0:
    #     num -= i
    #     i += 2
    # 
    # return num == 0
    
    
#给定一个整数，写一个函数来判断它是否是 3 的幂次方。
import sys

def isPowerOfThree(self, n: int) -> bool:
    '''首先想到常见的循环或者递归
    又因为是整数，这就限制了范围=>直接使用查找表也很快'''
    maxint = sys.maxsize
    return n > 0 and 1162261467 % n == 0

#238. 除自身以外数组的乘积
def productExceptSelf(self, nums: List[int]) -> List[int]:
    '''最先想到的是先把总乘积算出来，再依次相除，但是题目要求不允许使用除法
    暴力解法：依次相乘，毫无算法思维，舍弃之
    优化解法：和数组前缀和相似'''
    # n=len(nums)
    # output=[1]*n
    # output2=[1]*n
    # for i in range(1,n):
    #     output[i]=output[i-1]*nums[i-1]
    # for i in range(n-1,0,-1):
    #      output2[i-1]=output2[i]*nums[i]
    # output=[i*j for i,j in zip(output,output2)]
    # return output
    n = len(nums)
    output = [1] * n
    left = right = 1
    for i in range(1, n):
        left *= nums[i - 1]
        output[i] = left
    for i in range(n - 1, 0, -1):
        right *= nums[i]
        output[i - 1] *= right
    return output

#628. 三个数的最大乘积:在数组中找出由三个数组成的最大乘积，并输出这个乘积。
def maximumProduct(self, nums: List[int]) -> int:
    '''思路：数组问题+最大值问题：最小的两个负数和最大的一个正数或者最大的3个正数
    转化为找到这5个数=>排序或者线性扫描即可'''
    nums.sort()
    return max(nums[0] * nums[1] * nums[-1], nums[-1] * nums[-2] * nums[-3])

```

