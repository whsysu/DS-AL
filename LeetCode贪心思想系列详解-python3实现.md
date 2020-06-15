#### leetcode 贪心思想：保证每次操作都是局部最优的，并且最后得到的结果是全局最优的。

```python
#455. 分发饼干：每个孩子最多只能给一块饼干。孩子 i ，有胃口值 gi(满足胃口的饼干的最小尺寸)饼干 j ，尺寸 sj 。
# 如果 sj >= gi ，饼干 j 分配给孩子 i ，这个孩子会得到满足。尽可能满足越多数量的孩子，输出这个最大数值。
def findContentChildren(self, g, s) -> int:
    '''思路：最值问题，但是不是动态规划=>贪心思想
            贪心保障每一步最优:对g和s排序，大的分给大的，小的分给小的
        思路:双指针+贪心'''
    if not g and not s:
        return 0
    g.sort()
    s.sort()
    result = 0
    i, j = len(g) - 1, len(s) - 1
    while i >= 0 and j >= 0:
        if s[j] >= g[i]:
            result += 1
            i, j = i - 1, j - 1
        else:
            i -= 1
    return result


#435.贪心算法之区间调度问题：计算让一组区间不重叠所需要移除的区间个数
def eraseOverlapIntervals(self, intervals) -> int:
    '''按end排序'''
    if not intervals:
        return 0
    intervals = sorted(intervals, key=lambda x: x[1])
    result = 1
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] >= end:
            end = intervals[i][1]  # 因为是按照end排序的
            result += 1
    return len(intervals) - result


#452. 气球在一个水平数轴上摆放，可重叠，飞镖垂直投向坐标轴，使得路径上的气球都被刺破。求最小投飞镖数使所有气球被刺破。
def findMinArrowShots(self, points) -> int:
    '''重叠区间'''
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    result=1
    end=points[0][1]
    for i in range(1,len(points)):
        if points[i][0]>end:
            result+=1
            end=points[i][1]
    return result


#121. 买卖股票的最佳时机。只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
'''动态规划dp[i]表示在第i天卖出最大利润'''
def maxProfit(self, prices) -> int:
    if not prices:
        return 0
    dp=[0]*len(prices)
    buy=prices[0]
    for i in range(1,len(prices)):
        if prices[i-1]<buy:
            buy=prices[i-1]
        dp[i]=prices[i]-buy
    return max(dp)
# 再优化，因为没有递推关系式，所以直接维护buy和sell二个变量即可
def maxProfit(self, prices) -> int:
    if not prices:
        return 0
    buy, sell = prices[0], 0
    for i in range(1, len(prices)):
        if prices[i - 1] < buy:
            buy = prices[i - 1]
        if prices[i] - buy > sell:
            sell = prices[i] - buy
    return sell


#122. 买卖股票的最佳时机:交易次数不限
def maxProfit(self, prices) -> int:
    '''思考：首先这是最大最小值问题，属于动态规划系列'''
    # 第一步：画图
    # 定义状态 天数i，当前持有状态[0]代表未持有[1]代表持有 ； 选择【买，卖，不变】
    # 定义dp[i][0/1]含义 到第i天持有状态为[0/1]获得的最大利润
    # 边界条件
    if not prices:
        return 0
    n = len(prices)
    dp = [[0 for i in range(2)] for j in range(n)]  # dp[0]已经包含base case
    # base case
    dp[0][0], dp[0][1] = 0, -prices[0]
    for i in range(1, n):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        dp[i][1] = max(dp[i - 1][0] - prices[i], dp[i - 1][1])
    return dp[n - 1][0]
#贪心算法
def maxProfit(self, prices) -> int:
    profit = 0
    for i in range(1, len(prices)):
        tmp = prices[i] - prices[i - 1]
        if tmp > 0: profit += tmp
    return profit


#605. 种花问题
def canPlaceFlowers(self, flowerbed, n: int) -> bool:
    '''贪心想法：连续3个0，中间位置种花。对于数组的第一个和最后一个位置，我们只需要考虑一侧是否为 0'''
    count=0
    length=len(flowerbed)
    for i in range(length):
        if flowerbed[i]==0 and (i==0 or flowerbed[i-1]==0) and (i==length-1 or flowerbed[i+1]==0):
            flowerbed[i]=1
            count+=1
    return count>=n


#392. 判断 s 是否为 t 的子序列。s 和 t 中仅包含英文小写字母。字符串 t 可能会很长而 s 是个短字符串
def isSubsequence(self, s: str, t: str) -> bool:
    '''思路：字符串问题：常用双指针和动态规划'''
    # 和分饼干问题很相似啊，分饼干是大于等于，这里是等于
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)


#665. 最多 改变 1 个元素的情况下，该数组能否变成一个非递减数列。
'''在出现 nums[i] < nums[i - 1] 时，需要考虑的是应该修改数组的哪个数，使得本次修改能使 i 之前的数组成为非递减数组，
并且 不影响后续的操作 =>要分情况讨论'''
def checkPossibility(self, nums) -> bool:
    '''思路：整个数组一定是整体有序的
    依次遍历每个地方，判断是否是非递减=>过于重复
    其实只要判断是否有二个及以上的位置顺序有错即可'''
    count=0
    for i in range(1,len(nums)):
        if nums[i-1]<=nums[i]:
            continue
        if i-2>=0 and nums[i]<nums[i-2]:
            nums[i]=nums[i-1]
        else:
            nums[i-1]=nums[i]
        count+=1
    return count<=1


#53. 最大子序和
def maxSubArray(self, nums) -> int:
    '''思考：最大值系列，首先想到动态规划'''
    #dp[i]表示以nums[i]结尾的最大子数组的和=>由于dp[i]仅仅和前一个有关，所以可以再优化
    if not nums:
        return 0
    dp=[0]*len(nums)
    dp[0]=nums[0]
    for i in range(1,len(nums)):
        if dp[i-1]>0:
            dp[i]=dp[i-1]+nums[i]
        else:
            dp[i]=nums[i]
    return max(dp)


#763. S 由小写字母组成。把s划分为尽可能多的片段，同一个字母只会出现在其中的一个片段。返回一个表示每个字符串片段的长度的列表。
def partitionLabels(self, S: str):
    '''首先：出现最大值，首先想到动态规划，但是这道题貌似更符合贪心
    本质为区间规划问题：每个字母的首尾字母组成一条线，线的坐标为index；
    从第一条线开始，这样就不相交的区间有多少：等价于LeetCode435
    '''
    last = {char: index for index, char in enumerate(S)}  # 十分巧妙
    first = {}
    for char in last.keys():
        first[char] = S.index(char)
    num = []
    for a, b in zip(first.values(), last.values()):
        num.append([a, b])
    #注意num是按起始位置有序的
    end = num[0][1]
    result = []
    for i in range(1, len(num)):
        if num[i][0] > end:
            result.append(end + 1 - sum(result))
            end = num[i][1]
        else:
            end = max(end, num[i][1])
    result.append(len(S) - sum(result))
    return result
```

