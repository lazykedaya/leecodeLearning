import java.awt.*;
import java.awt.event.HierarchyBoundsAdapter;
import java.awt.font.NumericShaper;
import java.util.*;
import java.util.List;

public class graph_test {
    public static void main(String[] args) {
        int[][] a = {{4, 0},
                {0, 1},
                {1, 0},
                {2, 3}};
        int[][] b = {};
        System.out.println(canFinish(1, b));

    }
    //蛇梯棋
    public static int snakesAndLadders(int[][] board) {
        return 0;
    }

    // 课程表I,II
    public static List<List<Integer>> edges;//各门课指向
    public static int[] indeg;//各门课入度，两者长度为需求课程长度
    public static int[] visited;
    public static int[] result;
    public static boolean valid = true;
    public static int index;
    public static int[] canFinish(int numCourses, int[][] prerequisites) {
        //广度优先拓扑排序（从入度考虑，入度为0表示不需要前置学科）
        /*edges=new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            edges.add(new ArrayList<Integer>());
        }
        indeg=new int[numCourses];
        //对每门课的入度和指向进行初始化
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
            indeg[info[0]]++;
        }
        LinkedList<Integer> queue = new LinkedList<>();
        //如果哪个结点入度为0，则将其添加到队列中，表示这门课可以拿出来学了
        for (int i = 0; i < numCourses; i++) {
            if(indeg[i]==0)queue.offer(i);
        }
        //统计变量
        int visited=0;
        while (!queue.isEmpty()){
            visited++;
            int u=queue.poll();
            for (Integer v : edges.get(u)) {
                indeg[v]--;
                if(indeg[v]==0){
                    queue.offer(v);
                }
            }
        }
        //如果预修数组为空，那么表示每门课可以单独学，indeg一开始就全是0，queue里面自然全是0，visited最后也等于numCourses
        return visited==numCourses;*/
        //深度优先，以出度考虑,把每个节点分为未搜索，已搜索，在搜索
        edges = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            edges.add(new ArrayList<>());
        }
        visited = new int[numCourses];
        result = new int[numCourses];
        index = numCourses - 1;
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
        }
        for (int i = 0; i < numCourses && valid == true; i++) {
            //未搜索就进行深度搜索
            if (visited[i] == 0) {
                dfs3(i);
            }
        }
        if (!valid) return new int[0];
        return result;
    }
    private static void dfs3(int u) {
        //正在搜索
        visited[u] = 1;
        for (Integer v : edges.get(u)) {
            if (visited[v] == 0) {
                dfs3(v);
                if (!valid) return;
            }
            //碰到正在搜索表示有环
            else if (visited[v] == 1) {
                valid = false;
                return;
            }
        }
        visited[u] = 2;
        result[index--] = u;
    }

    // 除法求值
    public static double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        HashMap<String, Integer> variables = new HashMap<>();
        int index = 0;
        //先将所有字符存起来并对应一个数字，作为索引大小
        for (List<String> equation : equations) {
            for (String s : equation) {
                if (!variables.containsKey(s)) {
                    variables.put(s, index++);
                }
            }
        }
        // 对于每个点，存储其直接连接到的所有点及对应的权值
        List<Pair>[] edges = new List[index];
        for (int i = 0; i < index; i++) {
            edges[i] = new ArrayList<>();
        }
        int n = equations.size();
        //类似二维数组
        for (int i = 0; i < n; i++) {
            int va = variables.get(equations.get(i).get(0));
            int vb = variables.get(equations.get(i).get(1));
            edges[va].add(new Pair(vb, values[i]));
            edges[vb].add(new Pair(va, 1 / values[i]));
        }
        int size = queries.size();
        double[] ret = new double[size];
        for (int i = 0; i < size; i++) {
            List<String> query = queries.get(i);
            double ans = -1.0;
            if (variables.containsKey(query.get(0)) && variables.containsKey(query.get(1))) {
                //获取变量索引
                int ia = variables.get(query.get(0)), ib = variables.get(query.get(1));
                if (ia == ib) ans = 1.0;
                else {
                    LinkedList<Integer> points = new LinkedList<>();
                    points.offer(ia);
                    double[] ratio = new double[index];
                    Arrays.fill(ratio, -1.0);
                    ratio[ia] = 1.0;
                    while (!points.isEmpty() && ratio[ib] < 0) {
                        int x = points.poll();
                        //以x索引指向变量开头的所有结果遍历
                        for (Pair pair : edges[x]) {
                            int y = pair.index;
                            double val = pair.value;
                            if (ratio[y] < 0) {
                                ratio[y] = ratio[x] * val;
                                points.offer(y);
                            }
                        }
                    }
                    ans = ratio[ib];
                }
            }
            ret[i] = ans;
        }
        return ret;
    }
    public static class Pair {
        int index;
        double value;

        Pair(int index, double value) {
            this.value = value;
            this.index = index;
        }
    }

    //回溯 组合[1-n]取k个数
    public static List<Integer> temp = new ArrayList<Integer>();
    public static List<List<Integer>> ans = new ArrayList<List<Integer>>();
    public static List<List<Integer>> combine(int n, int k) {
        ArrayList<List<Integer>> combinations = new ArrayList<>();
        dfs2(1, n, k);
        return ans;
    }
    private static void dfs2(int cur, int n, int k) {
        if (temp.size() + (n - cur + 1) < k) return;
        // 记录合法的答案
        if (temp.size() == k) {
            ans.add(new ArrayList<>(temp));
            return;
        }
        // 考虑选择当前位置
        temp.add(cur);
        dfs2(cur + 1, n, k);
        temp.remove(temp.size() - 1);
        // 考虑不选择当前位置
        dfs2(cur + 1, n, k);
    }

    //回溯电话号码的字母组合
    public static List<String> letterCombinations(String digits) {
        ArrayList<String> combinations = new ArrayList<>();
        if (digits.length() == 0) return combinations;
        HashMap<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");
        backtrack(combinations, map, digits, 0, new StringBuffer());
        return combinations;
    }
    private static void backtrack(ArrayList<String> combinations, HashMap<Character, String> map, String digits, int index, StringBuffer combination) {
        if (index == digits.length()) {
            combinations.add(combination.toString());
        } else {
            char digit = digits.charAt(index);
            String letters = map.get(digit);
            int count = letters.length();
            for (int i = 0; i < count; i++) {
                combination.append(letters.charAt(i));
                backtrack(combinations, map, digits, index + 1, combination);
                combination.deleteCharAt(index);
            }
        }
    }

    // 克隆图
    public static HashMap<Node, Node> visited2 = new HashMap<>();
    public static Node cloneGraph(Node node) {
 /*       if (node == null) {
            return node;
        }
        // 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
        if (visited.containsKey(node)) {
            return visited.get(node);
        }
        // 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
        Node cloneNode = new Node(node.val, new ArrayList());
        // 哈希表存储
        visited.put(node, cloneNode);

        // 遍历该节点的邻居并更新克隆节点的邻居列表
        for (Node neighbor: node.neighbors) {
            cloneNode.neighbors.add(cloneGraph(neighbor));
        }
        return cloneNode;*/
        return node;
    }

    // 被围绕的区域
    public static void solve(char[][] board) {
        if (board == null || board.length <= 2) return;
        int r = board.length, c = board[0].length;
        //这个方法太复杂了，耗时比较长
       /* LinkedList<LinkedList<Integer>> lists = new LinkedList<>();
        for (int i = 1; i < r - 1; i++) {
            for (int j = 1; j < c - 1; j++) {
                if (board[i][j] == 'O') {
                    LinkedList<Integer> id = new LinkedList<>();
                    LinkedList<Integer> id2 = new LinkedList<>();
                    id.add(i * c + j);
                    id2.add(i * c + j);
                    board[i][j] = 'X';
                    boolean flag=false;
                    while (!id.isEmpty()) {
                        int x_y = id.remove();
                        int row = x_y / c;
                        int cow = x_y % c;
                        if (row - 1 >= 0 && board[row - 1][cow] == 'O') {
                            if(row-1==0)flag=true;
                            board[row - 1][cow] = 'X';
                            id.add((row - 1) * c + cow);
                            id2.add((row - 1) * c + cow);
                        }
                        if (row + 1 < r && board[row + 1][cow] == 'O') {
                            if(row+1==r-1)flag=true;
                            board[row + 1][cow] = 'X';
                            id.add((row + 1) * c + cow);
                            id2.add((row + 1) * c + cow);
                        }
                        if (cow - 1 >= 0 && board[row][cow - 1] == 'O') {
                            if(cow-1==0)flag=true;
                            board[row][cow - 1] = 'X';
                            id.add(row * c + cow - 1);
                            id2.add(row * c + cow - 1);
                        }
                        if (cow + 1 < c && board[row][cow + 1] == 'O') {
                            if(cow+1==r-1)flag=true;
                            board[row][cow + 1] = 'X';
                            id.add(row * c + cow + 1);
                            id2.add(row * c + cow + 1);
                        }
                    }
                    if(flag){
                        lists.add(id2);
                    }
                }
            }
        }
        while (!lists.isEmpty()){
            for (Integer id : lists.remove()) {
                int r1=id/c;
                int c1=id%c;
                board[r1][c1]='O';
            }
        }*/
        //先对边界进行预处理，把边界为‘O'并基于此’O'向内延伸的‘o’都替换为1，最后遍历一次，使O->X,1->O

    }

    // 岛屿数量,有广度优先和深度优先算法，dfs递归解法算一般性解法
    public static int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int m = grid.length, n = grid[0].length;
        int count = 0;
        //深度优先搜索（每次找到1，计数加1，并沿此把相连的1全标为0）
        /*
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(grid[i][j]=='1'){
                    count++;
                    dfs(grid,i,j);
                }
            }
        }
        return count;*/
        //广度优先搜索
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    grid[i][j] = '0';
                    LinkedList<Integer> neighbors = new LinkedList<>();
                    neighbors.add(i * n + j);
                    while (!neighbors.isEmpty()) {
                        int id = neighbors.remove();
                        int row = id / n;
                        int cow = id % n;
                        if (row - 1 >= 0 && grid[row - 1][cow] == '1') {
                            grid[row - 1][cow] = '0';
                            neighbors.add((row - 1) * n + cow);
                        }
                        if (row + 1 < m && grid[row + 1][cow] == '1') {
                            grid[row + 1][cow] = '0';
                            neighbors.add((row + 1) * n + cow);
                        }
                        if (cow - 1 >= 0 && grid[row][cow - 1] == '1') {
                            grid[row][cow - 1] = '0';
                            neighbors.add(row * n + cow - 1);
                        }
                        if (cow + 1 < n && grid[row][cow + 1] == '1') {
                            grid[row][cow + 1] = '0';
                            neighbors.add(row * n + cow + 1);
                        }
                    }
                }
            }
        }
        return count;
    }
    public static void dfs(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;
        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') return;
        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

}
