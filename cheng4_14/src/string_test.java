import com.sun.nio.sctp.SctpStandardSocketOptions;
import org.w3c.dom.Node;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.Instant;
import java.util.*;

public class string_test {
    public static void main(String[] args) {

        int left = 2147483646, right = 2147483647;
        /*int ans=left;
        for (int i = left+1; i <= right; i++) {
            ans =ans&i;
            if(ans==0)ans=0;
            if(i==right)return;
        }*/
        int[][] a = {{0, 1}, {0, 0}};
        int[] b = {1, 2};
        System.out.println(maxProfit(b));
        System.out.println(isInterleave("aabcc", "dbbca", "aadbbcbcac"));

    }

    //最大正方形
    public static int maximalSquare(char[][] matrix) {
        int maxSide = 0;
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return 0;
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i - 1][j - 1], dp[i][j - 1])) + 1;
                }
                maxSide = Math.max(maxSide, dp[i][j]);
            }
        }
        return (maxSide * maxSide);
    }

    //编辑距离
    public static int minDistance(String word1, String word2) {
        int n = word1.length(), m = word2.length();
        if (m * n == 0) return n + m;
        int[][] dp = new int[n + 1][m + 1];
        //边界状态
        for (int i = 0; i <= n; i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= m; i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int left = dp[i - 1][j] + 1;
                int down = dp[i][j - 1] + 1;
                int left_down = dp[i - 1][j - 1];
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    left_down += 1;
                }
                dp[i][j] = Math.min(left, Math.min(left_down, down));
            }
        }
        return dp[n][m];
    }

    //交错字符串
    public static boolean isInterleave(String s1, String s2, String s3) {
        int len1 = s1.length(), len2 = s2.length(), len3 = s3.length();
        if ((len1 + len2) != len3) return false;
        boolean[] f = new boolean[len2 + 1];
        f[0] = true;
        for (int i = 0; i <= len1; i++) {
            for (int j = 0; j <= len2; j++) {
                int p = i + j - 1;
                if (i > 0) {
                    f[j] = f[j] && s1.charAt(i - 1) == s3.charAt(p);
                }
                if (j > 0) {
                    f[j] = f[j] || (f[j - 1] && s2.charAt(j - 1) == s3.charAt(p));
                }
            }
        }
        return f[len2];
    }

    //买卖股票的最佳时机 4
    public static int maxProfit(int k, int[] prices) {
        if (prices.length == 0) return 0;
        int n = prices.length;
        k = Math.min(k, n / 2);
        int[] sell_i = new int[k + 1];
        int[] buy_i = new int[k + 1];
        buy_i[0] = -prices[0];
        sell_i[0] = 0;
        for (int i = 1; i <= k; i++) {
            buy_i[i] = sell_i[i] = Integer.MIN_VALUE / 2;
        }
        for (int i = 1; i < n; i++) {
            buy_i[0] = Math.max(buy_i[0], sell_i[0] - prices[i]);
            for (int j = 1; j <= k; j++) {
                buy_i[j] = Math.max(buy_i[j], sell_i[j] - prices[i]);
                sell_i[j] = Math.max(sell_i[j], buy_i[j - 1] + prices[i]);
            }
        }
        return Arrays.stream(sell_i).max().getAsInt();
    }

    //买卖股票的最佳时机 III
    public static int maxProfit(int[] prices) {
        /*int min=prices[0];
        int max1=0,max2=0;
        for (int i = 1; i < prices.length; i++) {
            if(prices[i]<=min){min=prices[i];continue;}
            int temp=0;
            while (i<prices.length&&(prices[i]-min)>=max1){
                temp=Math.max(temp,prices[i]-min);
                i++;
            }
            if(temp>=max1){
                max2=max1;
                max1=temp;
            }else if(temp>=max2){
                max2=temp;
            }
            if((i+1)<prices.length){
                min=prices[i+1];
            }
        }
        return max2+max1;*/
        //思路就是把数组分成两部分，看前面部分和后面部分的最大值的和，并对每个和进行比对
        //直接搞会超出时间限制，需要将部分结果存储起来
        /*int max=0;
        int min1;int min2;
        for (int i = 1; i < prices.length; i++) {
            min1=prices[0];int max1=0,max2=0;
            for (int j = 0; j <= i; j++) {
                if(prices[j]>min1){
                    max1= Math.max(max1,prices[j]-min1);
                }else min1=prices[j];
            }
            min2=prices[i];
            for (int k = i; k < prices.length; k++) {
                if(prices[k]>min2){
                    max2= Math.max(max2,prices[k]-min2);
                }else min2=prices[k];
            }
            max= Math.max(max,max1+max2);
        }
        return max;*/
        int len = prices.length;
        int min1 = prices[0], max1 = prices[len - 1];
        int max = 0;
        int[] zh = new int[len];
        int[] fu = new int[len];
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] > min1) {
                max = Math.max(max, prices[i] - min1);
            } else min1 = prices[i];
            zh[i] = max;
        }
        int max2 = 0;
        for (int i = len - 1; i >= 0; i--) {
            if (prices[i] < max1) {
                max2 = Math.max(max2, max1 - prices[i]);
            } else max1 = prices[i];
            fu[i] = max2;
        }
        int ans = 0;
        for (int i = 0; i < len; i++) {
            ans = Math.max(ans, zh[i] + fu[i]);
        }
        return ans;
    }

    //最长回文子串
    public static String longestPalindrome(String s) {
        //中心扩展算法，假设当前字母为回文中心，并进行拓展
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expendAroundCenter(s, i, i);
            int len2 = expendAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    //回文中心扩展长度
    private static int expendAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    //不同路径 II
    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[][] ans = new int[m][n];
        if (obstacleGrid[0][0] == 1) return 0;
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) break;
            else ans[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1) break;
            else ans[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] != 1) ans[i][j] = ans[i - 1][j] + ans[i][j - 1];
            }
        }
        return ans[m - 1][n - 1];
    }

    //最小路径和
    public static int minPathSum(int[][] grid) {
        int m = grid.length - 1, n = grid[0].length - 1;
        /* return minSumFu1(grid, m, n);*/
        for (int i = 1; i <= n; i++) {
            grid[0][i] += grid[0][i - 1];
        }
        for (int i = 1; i <= m; i++) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[m][n];
    }

    public static int minSumFu1(int[][] grid, int m, int n) {
        //超出时间限制了
        if (m == 0) {
            int ans = 0;
            for (int i = 0; i <= n; i++) {
                ans += grid[0][i];
            }
            return ans;
        }
        if (n == 0) {
            int ans = 0;
            for (int i = 0; i <= m; i++) {
                ans += grid[i][0];
            }
            return ans;
        }
        return Math.min(minSumFu1(grid, m - 1, n), minSumFu1(grid, m, n - 1)) + grid[m][n];
    }

    //只出现一次的数字 II
    public static int singleNumber(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        int ans = -1;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() == 1)
                ans = entry.getKey();
        }
        return ans;
        //位运算解法,//出现三次


    }

    //二进制求和
    public static String addBinary(String a, String b) {
        //转成10进制求和后
        int sum1 = 0, sum2 = 0;
        int index1 = 0, index2 = 0;
        for (int i = a.length() - 1; i >= 0; i--) {
            sum1 += (a.charAt(i) - '0') * Math.pow(2, index1);
            index1++;
        }
        for (int j = b.length() - 1; j >= 0; j--) {
            sum2 += (b.charAt(j) - '0') * Math.pow(2, index2);
            index2++;
        }
        StringBuffer res = new StringBuffer();
        int num = sum1 + sum2;
        if (num == 0) return "0";
        while (num != 0) {
            res.append(num % 2);
            num /= 2;
        }
        return res.reverse().toString();
       /* 模拟
       //   StringBuffer ans = new StringBuffer();
        //
        //        int n = Math.max(a.length(), b.length()), carry = 0;
        //        for (int i = 0; i < n; ++i) {
        //            carry += i < a.length() ? (a.charAt(a.length() - 1 - i) - '0') : 0;
        //            carry += i < b.length() ? (b.charAt(b.length() - 1 - i) - '0') : 0;
        //            ans.append((char) (carry % 2 + '0'));
        //            carry /= 2;
        //        }
        //
        //        if (carry > 0) {
        //            ans.append('1');
        //        }
        //        ans.reverse();
        //
        //        return ans.toString();*/


    }

    //单词拼接
    public static boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        // dp[i]表示s前i个字符能由wordDict里面的单词拼成；
        //因此只需判断dp[i]=dp[j]&&set.set.contains(s.substring(j,i))
        //知道i等于s长度的时候看dp[s.length]是否为true即可
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    //加油站
    public static int canCompleteCircuit(int[] gas, int[] cost) {
        int len = gas.length;
        int i = 0;
        while (i < len) {
            int sumOfGas = 0, sumOfCost = 0;
            int cnt = 0;
            while (cnt < len) {
                int j = (i + cnt) % len;
                sumOfGas += gas[j];
                sumOfCost += cost[j];
                if (sumOfCost > sumOfGas) {
                    break;
                }
                cnt++;
            }
            if (cnt == len) {
                return 1;
            } else {
                i = i + cnt + 1;
            }
        }
        return -1;
       /* int sum = 0;
        int min = Integer.MAX_VALUE;
        int minIndex = -1;
        for(int i = 0; i < gas.length; i++){
            sum = sum + gas[i] - cost[i];
            if(sum < min && sum < 0){
                min = sum;
                minIndex = i;
            }
        }
        if(sum < 0) return -1;
        return (minIndex + 1 )%gas.length;*/


    }

    //子序列 动态规划*****
    public static boolean isSubsequence(String s, String t) {
        /*int len = t.length();
        int lenS = s.length();
        int index=0;
        for (int i = 0; i < len; i++) {
            if(index<lenS&&t.charAt(i)==s.charAt(index)){
                index++;
            }
            if(index==lenS)return true;
        }
        return false;*/
        //对于大量判断序列，使用动态规划
        //先对t进行预处理
        int len = t.length();
        int lenS = s.length();
        int[][] pre = new int[len + 1][26];
        //二维数组中26表示26个字母 令 pre[i][j]表示字符串t中从位置i开始往后字符j第一次出现的位置
        for (int i = 0; i < 26; i++) {
            pre[len][i] = len;
        }
        for (int i = len - 1; i >= 0; i--) {
            for (int j = 0; j < 26; j++) {
                if (t.charAt(i) == j + 'a')
                    pre[i][j] = i;
                else pre[i][j] = pre[i + 1][j];
            }
        }
        int add = 0;
        for (int i = 0; i < lenS; i++) {
            if (pre[add][s.charAt(i) - 'a'] == len)
                return false;
            add = pre[add][s.charAt(i) - 'a'] + 1;
        }
        return true;
    }

    //1验证回文串---字符串的处理，正则表达，可变字符串；
    public static boolean isPalindrome(String s) {
        //用stringBuffer来存储字符，后面用stringBuffer.reverse().toString()与之前的比较
        //双指针
        String s1 = s.toLowerCase();
        //判断是否是字母
        //Character.isLetterOrDigit(s.charAt())
        //大小写也可以单个进行
        //Character.toLowerCase()
        String s2 = s1.replaceAll("[^0-9a-z]", "");
        byte[] bytes = s2.getBytes();
        int L = 0;
        int len = bytes.length;
        int R = len - 1;
        while (L < R) {
            if (bytes[L] == 32) L++;
            else if (bytes[R] == 32) R--;
            else if (bytes[R] != bytes[L]) return false;
            else if (bytes[R] == bytes[L]) {
                R--;
                L++;
            }
        }
        return true;
    }
}
