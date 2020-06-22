#### #哈希表

```python
#哈希表使用 O(N) 空间复杂度存储数据，并且以 O(1) 时间复杂度求解问题。
#HashMap 主要用于映射关系，从而把两个元素联系起来.python中hashmap为字典

#1. 两数之和：给定整数数组 nums 和目标值 target，在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
#思路1 先排序  则再找id   NlogN  大量数据不太好
#思路 依次遍历  变成查找另外一个数   查找最快的是hash hash记录每个元素及出现次数
#如果hash记录每个元素及id则更快
def twoSum(self, nums, target: int):
    dic={}
    n=len(nums)
    for i in range(0,n):
        dic[nums[i]]=i
    for i in range(0,n):
        k=dic.get(target-nums[i],-1)
        if k!=i and k!=-1:
            return [i,dic[target-nums[i]]]

#217. 存在重复元素：给定一个整数数组，判断是否存在重复元素。
def containsDuplicate(self, nums) -> bool:
    '''思路：使用set集合 判断len(nums)==len(set)
    使用hash记录元素出现次数，统计，一旦超过2,返回false
    '''
    #pass

#594. 最长和谐子序列：和谐数组是指一个数组里元素的最大值和最小值之间的差别正好是1。
#在所有可能的子序列中找到最长的和谐子序列的长度。
def findLHS(self, nums) -> int:
    dicts = {}
    for i in nums:
        dicts[i] = dicts.get(i, 0) + 1
    res = 0
    for i in dicts:
        if i + 1 in dicts:
            res = max(res, dicts[i] + dicts[i + 1])
    return res

#128. 最长连续序列:给定一个未排序的整数数组，找出最长连续序列的长度。要求算法的时间复杂度为 O(n)。
'''思路：对非排序数组，要求连续序列，又不能排序
        1.先排序，再统计：时间复杂度O(NlogN),不行
        2.要求时间复杂度为O(N):则只能遍历一次，然后记住关键信息：时间要求高那只能用空间来换：
        3.最大值问题：动态规划
        '''
def longestConsecutive(self, nums):
    longest_streak = 0
    num_set = set(nums)

    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)
    return longest_streak


```

