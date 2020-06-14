```python
#LeetCode双指针系列详解:主要用于遍历数组/字符串/链表=>顺序表，两个指针指向不同的元素，从而协同完成任务

#167 升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
def twoSum(self, numbers, target):
    # 首尾指针遍历
    l, h = 0, len(numbers) - 1
    while l < h:
        if numbers[l] + numbers[h] == target:
            return l + 1, h + 1
        elif numbers[l] + numbers[h] < target:
            l += 1
        else:
            h -= 1
    return [-1, -1]


#633，给定一个非负整数 c ，你要判断是否存在两个整数 a 和 b，使得 a2 + b2 = c。
def judgeSquareSum(self, c: int) -> bool:
    #思路：1=>sqrt(c)  的有序数组使用双指针
    num=[i for i in range(0,int(sqrt(c))+1)]
    i,j=0,len(num)-1
    while i<=j:
        if num[i]**2+num[j]**2==c:
            return True
        elif num[i]**2+num[j]**2>c:
            j-=1
        else:
            i+=1
    return False


#345. 反转字符串中的元音字母
def reverseVowels(self, s: str) -> str:
    '''首尾指针反转，类似于快速排序的partition函数'''
    if len(s)<=1:
        return s
    i,j=0,len(s)-1
    word=['a','o','e','u','i','A','O','E','U','I']
    while i<j:
        while i<j and s[i] not in word:
            i+=1
        while i<j and s[j] not in word:
            j-=1
        if i<j:
            s=s[0:i]+s[j]+s[i+1:j]+s[i]+s[j+1:]
            i,j=i+1,j-1
    return s


#680 给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
def isvalid(self, s):
    return s == s[::-1]
def validPalindrome(self, s: str) -> bool:
    i ,j= 0,len(s) - 1
    while i <= j:
        if s[i] == s[j]:
            i += 1
            j -= 1
        else:
            return self.isvalid(s[i + 1:j + 1]) or self.isvalid(s[i:j])

    return True


#88. 合并两个有序数组, 元素数量分别为 m 和 n .nums1 有足够的空间来保存 nums2 中的元素。
def merge(self, nums1, m, nums2, n):
    """
    Do not return anything, modify nums1 in-place instead.
    """
    '''其实就是归并排序的merge部分=>二个数组双指针'''
    i, j, k = m - 1, n - 1, m + n - 1
    while i >= 0 or j >= 0:
        if j == -1 or (i >= 0 and nums1[i] > nums2[j]):
            nums1[k] = nums1[i]
            i, k = i - 1, k - 1
        else:
            nums1[k] = nums2[j]
            j, k = j - 1, k - 1


#141给定一个链表，判断链表中是否有环。
def hasCycle(self, head) -> bool:
    '''思路1：用hash保存走过的节点，如果在hash中则有环
        思路2：双指针：fast走二步，low走一步，一定会相遇'''
    se = set()
    p = head
    while p:
        if p not in se:
            se.add(p)
        else:
            return True
        p = p.next
    return False
def hasCycle(self, head) -> bool:
    '''思路2：双指针：fast走二步，low走一步，一定会相遇'''
    if not head :
        return False
    fast, slow= head.next,head
    while fast and fast.next:
        if fast == slow:
            return True
        fast = fast.next.next
        slow = slow.next
    return False


#524. 通过删除字母匹配到字典里最长单词:找到字典里面最长的字符串，该字符串可以通过删除给定字符串的某些字符来得到。
#如果答案不止一个，返回长度最长且字典顺序最小的字符串。如果答案不存在，则返回空字符串。
def is_part(self, s, word):
    '''判断word是否为s的子序列'''
    i = j = 0
    while i < len(s) and j < len(word):
        if s[i] == word[j]:
            i, j = i + 1, j + 1
        else:
            i += 1
    return j == len(word)
def findLongestWord(self, s, d) -> str:
    '''遍历字典，判断word是否是s的子序列，找到最长的word即可'''
    length = 0
    result = ''
    for word in d:
        if self.is_part(s, word):
            if len(word) > length:
                length = len(word)
                result = word
            elif len(word) == length and word < result:
                result = word
    return result
```

