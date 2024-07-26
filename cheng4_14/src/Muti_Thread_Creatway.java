import java.util.ArrayList;

public class Muti_Thread_Creatway {
    /***
    Java中有几种方式创建线程
     继承thread，实现Runnable接口，实现Callable接口、
     ***/
    public static void main(String[] args) {
        new Thread(()-> System.out.println(Thread.currentThread().getName()+"abs")).start();
        new Thread(()-> System.out.println(Thread.currentThread().getName()+"abs1")).start();
        Thread thread = new Thread(() -> System.out.println(Thread.currentThread().getName() + "abs1"));
        System.out.println(thread.getState());

    }
}
