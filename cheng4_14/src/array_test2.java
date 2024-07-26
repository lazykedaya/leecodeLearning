import java.util.*;

public class array_test2 {
    public static void main(String[] args) {
        int[] a = {13, 17};
        int[][] nums1 = {{3, 5}, {4, 7}, {5, 10}, {12, 16}};
        System.out.println(findMinArrowShots(nums1));
        coinChange(new int[]{1, 2, 5}, 15);

    }

    //打家劫舍
    public static int rob(int[] nums) {
        //一维动态规划

        //每次走1或2步一共有多少种，再把每一种的结果比较
        //left表示f(k-2)right表示f(k-1)
        int left = 0, right = 0;
        for (int num : nums) {
            int temp = Math.max(right, left + num);
            left = right;
            right = temp;
        }
        return right;

    }

    //最长递增子序列
    public static int lengthOfLIS(int[] nums) {
        int len = 1, n = nums.length;
        int[] d = new int[n + 1];
        if (n == 0) return 0;
        d[len] = nums[0];
        for (int i = 1; i < n; i++) {
            if (nums[i] > d[len]) {
                d[++len] = nums[i];
            } else {
                int l = 1, r = len, pos = 0;
                while (l <= r) {
                    int mid = (l + r) / 2;
                    if (d[mid] < nums[i]) {
                        pos = mid;
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
                d[pos + 1] = nums[i];
            }
        }
        return len;
       /* int len=nums.length;
        if(len==0)return 0;
        int[] dp = new int[len];
        dp[0]=1;
        int maxans=1;
        for (int i = 0; i < len; i++) {
            dp[i]=1;
            for (int j = 0; j < i; j++) {
                if(nums[i]>nums[j]){
                    dp[i]= Math.max(dp[j]+1,dp[i]);
                }
            }
            maxans= Math.max(maxans,dp[i]);
        }
        return maxans;*/
    }

    //零钱兑换
    public static int coinChange(int[] coins, int amount) {
        /*if(amount<1)return 0;
        return coinChange(coins,amount,new int[amount]);*/
        int max = amount + 1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];

    }

    private static int coinChange(int[] coins, int rem, int[] count) {
        if (rem < 0) return -1;
        if (rem == 0) return 0;
        if (count[rem - 1] != 0) {
            return count[rem - 1];
        }
        int min = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = coinChange(coins, rem - coin, count);
            if (res >= 0 && res < min) {
                min = 1 + res;
            }
        }
        count[rem - 1] = (min == Integer.MAX_VALUE) ? -1 : min;
        return count[rem - 1];
    }

    //用最少数量的箭引爆气球
    public static int findMinArrowShots(int[][] points) {
        int len = points.length;
        //排序
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }
        });
        long left = (long) points[0][0], right = points[0][1];
        int count = 0;
        for (int i = 1; i < len; i++) {
            if (right < points[i][0]) {
                //没有交集
                left = points[i][0];
                right = points[i][1];
            } else {
                left = Math.max(left, points[i][0]);
                right = Math.min(right, points[i][1]);
                count++;
            }
        }
        return len - count;
    }

    //插入区间
    public static int[][] insert(int[][] intervals, int[] newInterval) {
       /* int len = intervals.length;
        if (len == 0) return new int[][]{newInterval};
        ArrayList<int[]> ans = new ArrayList<>();
        int a = newInterval[0], b = newInterval[1];
        if (b < intervals[0][0]) ans.add(newInterval);
        else if (b <= intervals[0][1] || (len - 1 > 0 && b < intervals[1][0])) {
            intervals[0][0] = Math.min(intervals[0][0], a);
            {
                intervals[0][1] = Math.max(intervals[0][1], b);
                a = intervals[0][0];
                b = intervals[0][1];
            }
        }
        for (int i = 0; i < len; i++) {
            if (a > intervals[i][1] || b < intervals[i][0]) {
                ans.add(intervals[i]);
            } else {
                ans.add(new int[]{Math.min(a, intervals[i][0]),0});
                for (int j = i; j < len; j++) {
                    if (b < intervals[j][0]) {
                        ans.get(i)[1] = b;
                        i = j - 1;
                        break;
                    } else if (b <= intervals[j][1] || j == len - 1) {
                        ans.get(i)[1] = Math.max(b, intervals[j][1]);
                        i = j;
                        break;
                    }
                }

            }
        }
        if(a>intervals[len-1][1])ans.add(newInterval);
        return ans.toArray(new int[][]{});*/
        //有几种情况1，比开头小2，右边与1区间有交集3，左右有交集4，左右无交集在中间5，在结尾有两种
        int left = newInterval[0];
        int right = newInterval[1];
        boolean placed = false;
        List<int[]> ansList = new ArrayList<int[]>();
        for (int[] interval : intervals) {
            if (interval[0] > right) {
                // 在插入区间的右侧且无交集
                if (!placed) {
                    ansList.add(new int[]{left, right});
                    placed = true;
                }
                ansList.add(interval);
            } else if (interval[1] < left) {
                // 在插入区间的左侧且无交集
                ansList.add(interval);
            } else {
                // 与插入区间有交集，计算它们的并集
                left = Math.min(left, interval[0]);
                right = Math.max(right, interval[1]);
            }
        }
        if (!placed) {
            ansList.add(new int[]{left, right});
        }
        return ansList.toArray(new int[][]{});
    }

    // 合并区间
    public static int[][] merge(int[][] intervals) {
        int m = intervals.length;
        ArrayList<int[]> ans = new ArrayList<>();
        //需要先排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        int min = intervals[0][0], max = intervals[0][1];
        for (int i = 1; i < m; i++) {
            int a = intervals[i][0], b = intervals[i][1];
            if (a >= min && a <= max) max = Math.max(max, b);
            else {
                ans.add(new int[]{min, max});
                min = a;
                max = b;
            }
        }
        ans.add(new int[]{min, max});
        return ans.toArray(new int[][]{});
    }

    // 生命游戏
    public static void gameOfLife(int[][] board) {
        //如果要在原数组更改，可以引入状态2和-1,2表示之前是活的但是更新后死了，-1则相反，这样就不会影响后面的更新了
        int m = board.length, n = board[0].length;
        int[][] ans = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int count = 0;
                //判断八个方向的数可以使用{-1,0,1}数组，进行3x3的循环（除去原点（自身））
                if (i - 1 >= 0 && board[i - 1][j] == 1) count++;
                if (i + 1 < m && board[i + 1][j] == 1) count++;
                if (j + 1 < n && board[i][j + 1] == 1) count++;
                if (j - 1 >= 0 && board[i][j - 1] == 1) count++;
                if (j - 1 >= 0 && i - 1 >= 0 && board[i - 1][j - 1] == 1) count++;
                if (j - 1 >= 0 && i + 1 < m && board[i + 1][j - 1] == 1) count++;
                if (j + 1 < n && i + 1 < m && board[i + 1][j + 1] == 1) count++;
                if (j + 1 < n && i - 1 >= 0 && board[i - 1][j + 1] == 1) count++;
                int temp = board[i][j];
                if (temp == 1) {
                    if (count == 2 || count == 3) ans[i][j] = 1;
                    else ans[i][j] = 0;
                } else {
                    if (count == 3) ans[i][j] = 1;
                    else ans[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = ans[i][j];
            }
        }
    }

    //矩阵置零
    public static void setZeroes(int[][] matrix) {
        //得先找到0
        int row = matrix.length;
        int cow = matrix[0].length;
        int[] m = new int[row];
        int[] n = new int[cow];
        //找出哪行哪列有0
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < cow; j++) {
                if (matrix[i][j] == 0) {
                    m[i]++;
                    n[j]++;
                }
            }
        }
        //把行填0
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < cow; j++) {
                if (m[i] > 0 || n[j] > 0) matrix[i][j] = 0;
            }
        }
    }

    //最长连续序列,要求时间复杂度为 O(n)
    public static int longestConsecutive(int[] nums) {
        //set会自动排序，去重
        //去重后从一边考虑，set遍历，当不存在比当前值小1的时候，找所有比当前值大的1，每次更新当前值和连续的长度
        int len = nums.length;
        Arrays.sort(nums);//[1,2,3,4,100]
        int max = 1;
        int temp = 1;
        if (len < 1) return 0;
        for (int i = 1; i < len; i++) {
            if ((nums[i] - nums[i - 1]) == 1) {
                temp++;
            } else if ((nums[i] - nums[i - 1]) == 0) continue;
            else temp = 1;
            max = Math.max(temp, max);
        }
        return max;
        /*Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }
        int longestStreak = 0;
        for (int num : num_set) {
            if (!num_set.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
                while (num_set.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }
                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }
        return longestStreak;
*/
    }

    //字母异位词分组
    public static List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            //这一步是关键1，将排好序的字符串作为键，同时char[]可以直接转为string
            String key = new String(chars);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);
            map.put(key, list);
        }
        //map可以获取所有键或所有值
        return new ArrayList<List<String>>(map.values());
    }

    //1快乐数
    public static boolean isHappy(int n) {
        //4→16→37→58→89→145→42→20→4。
        // 所有其他数字都在进入这个循环的链上，或者在进入 111 的链上
        /*int temp=n;
        HashSet<Integer> set = new HashSet<>();
        //能得到1的数为，1，10,100,1000,10000.。。。，找不到的时候怎么终止,有循环
        while (!set.contains(temp)){
            set.add(temp);
            int sum=0;
            while (temp!=0){
                int a = temp % 10;
                temp=temp/10;
                sum+=Math.pow(a,2);
            }
            if(sum==1)return true;
            temp=sum;
        }
        return false;*/
        //快慢指针
        int slow = n;
        int fast = getNext(n);
        while (fast != 1 && slow != fast) {
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }
        return fast == 1;

    }

    public static int getNext(int n) {
        int sum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            sum += d * d;
        }
        return sum;
    }

}
