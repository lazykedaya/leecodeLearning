import java.io.FileReader;
import java.util.HashMap;

public class LRUCache {
    ListNode LRU;
    ListNode tail;
    HashMap<Integer,Integer> map;
/***
* 以 正整数 作为容量 capacity 初始化 LRU 缓存*/
    public LRUCache(int capacity) {
        tail = new ListNode(-capacity-1);
        LRU=tail;
        map=new HashMap<>();
        for (int i = 1; i <= capacity-1; i++) {
            tail.next=new ListNode(-i);
            tail=tail.next;
            map.put(-i,-1);
        }

    }
/***
 * 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1
 ***/
    public int get(int key) {
        return map.getOrDefault(key, -1);
    }

    /***
    如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出最早存入的关键字。
    ***/
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            map.put(key,value);
        }else if(LRU!=null) {
            int a=LRU.val;
            map.remove(a);
            map.put(key,value);
            tail.next=new ListNode(key);
            tail=tail.next;
            LRU=LRU.next;
        }
    }
}
