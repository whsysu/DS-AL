#### LeetCode 二分查找

```python

#普通的二分查找
def binary_search(array,num):
    low,high=0,len(array)-1
    while low<=high:
        mid = (low + high) // 2
        if num==array[mid]:
            return True
        elif num<array[mid]:
            high=mid-1
        else:
            low=mid+1
    return False
#普通的二分查找递归写法
def binary_search_re(arr,target,low,high):
    if low<high:
        return False
    else:
        mid=(low+high)//2
        if target==arr[mid]:
            return True
        elif target<arr[mid]:
            return binary_search_re(arr,target,low,mid-1)
        else:
            return binary_search_re(arr,target,mid+1,high)


#69. x 的平方根
def mySqrt(self, x: int) -> int:
    '''思路：在[0-x]中查找y*y<x and (y+1)*(y+1)>x'''
    if x <= 0:
        return 0
    l, h = 0, x
    while l <= h:
        m = (l + h) // 2
        if m * m <= x and (m + 1) * (m + 1) > x:
            return m
        elif m * m > x:
            h = m - 1
        else:
            l = m + 1


#744. 寻找比目标字母大的最小字母：排序后的字符列表只包含小写英文字母。请你寻找在这一有序列表里比目标字母大的最小字母。
def nextGreatestLetter(self, letters, target: str) -> str:
    # 下面就是查找了，因为letters有序，首先想到二分查找
    l, h = 0, len(letters) - 1
    while l <= h:
        m = (l + h) // 2
        if letters[m] <= target:  # 若有重复，则查找最右边
            l = m + 1
        else:
            h = m - 1
    return letters[l] if l < len(letters) else letters[0]


#540. 有序数组中的单一元素：只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。
def singleNonDuplicate(self, nums) -> int:
    '''思路：输入有序则首先想到二分查找:nums一定是奇数
    此外：本题可以线性查找或者位运算，时间复杂度都是O(N)'''
    n = len(nums)
    l, h = 0, n - 1
    while l < h:
        m = (l + h) // 2
        if m % 2 == 1:
            m -= 1  # 使得m在偶数位置
        if nums[m] == nums[m + 1]:  # 原本在没有单个元素的情况下
            l = m + 2
        else:
            h = m
    return nums[l]


#278. 第一个错误的版本：假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
#通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错
def firstBadVersion(self, n):
    '''思路：版本之前有依赖，且为查找问题,首先想到二分查找'''
    l, h = 1, n
    while l < h:
        m = (l + h) // 2
        if isBadVersion(m):  # 查找左边界问题
            h = m
        else:
            l = m + 1

    return l


#153. 假设按照升序排序的数组在预先未知的某个点上进行了旋转。请找出其中最小的元素。可以假设数组中不存在重复元素。
def findMin(self, nums) -> int:
    '''
    部分有序数组中查找：首先想到二分查找+约束条件
    理想=完美   现实=理想+约束
    '''
    n=len(nums)
    if n==1 or nums[-1]>nums[0]: #没有发生旋转
        return nums[0]

    l,h=0,n-1
    while l<h:
        m=(l+h)//2
        if nums[m]>=nums[0]:
            l=m+1
        else:
            h=m
    return nums[l]


#34. 在排序数组中查找元素的第一个和最后一个位置
def findleft(self, nums, target):
    l, h = 0, len(nums) - 1
    while l <= h:
        m = (l + h) // 2
        if nums[m] == target:
            h = m - 1    #和普通搜索第一个不一样的地方
        elif nums[m] > target:
            h = m - 1
        elif nums[m] < target:
            l = m + 1
    if l >= len(nums) or nums[l] != target:  #和普通搜索第二个不一样的地方
        return -1
    return l

def findright(self, nums, target):
    l, h = 0, len(nums) - 1
    while l <= h:
        m = (l + h) // 2
        if nums[m] == target:
            l = m + 1
        elif nums[m] > target:
            h = m - 1
        elif nums[m] < target:
            l = m + 1
    if h < 0 or nums[h] != target:
        return -1
    return h

def searchRange(self, nums, target: int) :
    '''首先：输入有序，首先想到二分查找'''
    left = self.findleft(nums, target)
    right = self.findright(nums, target)
    return [left, right]

```

