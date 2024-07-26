import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class test {
    public static int a;
    public static void main(String[] args) {
        LRUCache lru = new LRUCache(2);
        lru.put(1,1);
        lru.put(2,2);
        lru.put(3,3);
        lru.get(2);

        a=2;
        sub1(a);
        System.out.println(a);
    }

    private static void sub1 (int b) {
        a=b-1;
    }

}
