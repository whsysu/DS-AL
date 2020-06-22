#### #字符串

```python
#字符串和数组类似，都是连续存储对象



#242. 有效的字母异位词：两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
# 思路1：排序后判断是否相等
# 思路二：由于只涉及到字母，可以用计数排序或者哈希表，判断二个hash是否相等
def isAnagram(self, s: str, t: str) -> bool:
    arr1 = list(s)
    arr1.sort()
    arr2 = list(t)
    arr2.sort()
    return arr1 == arr2

import collections
#409. 最长回文串：一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。
def longestPalindrome(self, s):
    ans = 0
    count = collections.Counter(s)
    for v in count.values():
        ans += v // 2 * 2
        if ans % 2 == 0 and v % 2 == 1:
            ans += 1
    return ans

#205. 同构字符串:如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。
#所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。
def isIsomorphic(self, s: str, t: str) -> bool:
    '''思路：为s中的每个字符串建立一个映射，但是映射必须相等'''
    if not s:
        return True
    dic = {}
    for i in range(len(s)):
        if s[i] not in dic:
            if t[i] in dic.values():
                return False
            else:
                dic[s[i]] = t[i]
        else:
            if dic[s[i]] != t[i]:
                return False
    return True

#647. 回文子串:计算这个字符串中有多少个回文子串。
#dp[i][j]定义为是否是回文子串，再求矩阵的和
def countSubstrings(self, s: str) -> int:
    n=len(s)
    if n<=0:
        return 0
    dp=[[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        dp[i][i]=1
    for j in range(1,n):
        for i in range(j-1,-1,-1):
            if s[j]==s[i]:
                if j-i<=2:
                    dp[i][j]=1
                else:
                    dp[i][j]=dp[i+1][j-1]
            else:
                dp[i][j]=0
    res=0
    for i in range(0,n):
        for j in range(0,n):
            res=res+dp[i][j]
    return res

#696. 计数二进制子串
#"00110011"=>6 #解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。
def countBinarySubstrings(self, s: str) -> int:
    n = len(s)
    if n <= 1:
        return 0
    ##记录当前重复字符串的个数
    curLen = 1
    preLen = 0
    ##记录子串的个数
    count = 0

    for i in range(1, n):
        if s[i] == s[i - 1]:
            curLen += 1
        else:
            preLen = curLen
            curLen = 1
        if preLen >= curLen:  ##preLen>=curLen:很重要，if only 等于，则只有0011的情况，当存在>情况时，就存在01这种情况。仔细读一下代码即可
            count += 1

    return count


```

