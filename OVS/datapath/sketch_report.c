#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include<linux/in.h>
#include<linux/inet.h>
#include<linux/socket.h>
#include<net/sock.h>
#include <linux/sched.h>
#include <linux/kthread.h>
#include "flow_key.h"
#define MAX_BATCH 1024
typedef int64_t elemtype;
unsigned short listen_port = 0x8888;


struct sketch_q {
    void* instance;
    elemtype(*query)(void* this, struct flow_key* key);
};

int get_stat_and_send_back(struct socket* client_sock, char* buf, int len, struct sketch_q* sketch) {
    if (len % sizeof(struct flow_key)) {
        printk("unexpected len: %d\n", len);
        return -1;
    }

    printk("querying...\n");
    int count = len / sizeof(struct flow_key);
    elemtype* sendbuf = kzalloc(count * sizeof(elemtype), GFP_KERNEL);
    struct flow_key* src = (struct flow_key*)buf;
    int i = 0;

    if(sketch==NULL||sketch->instance==NULL||sketch->query==NULL) {
        printk("no sketch associated\n");
        for (i = 0; i < count; ++i) {
            sendbuf[i] = i;
        }
    }
    else {
        for (i = 0; i < count; ++i) {
            sendbuf[i] = sketch->query(sketch->instance,&(src[i]));
        }
    }

    printk("query finished, sending\n");
    struct kvec vec;
    struct msghdr msg;
    memset(&vec, 0, sizeof(vec));
    memset(&msg, 0, sizeof(msg));
    vec.iov_base = sendbuf;
    vec.iov_len = count * sizeof(elemtype);

    kernel_sendmsg(client_sock, &msg, &vec, 1, count * sizeof(elemtype));
    kfree(sendbuf);
    return 0;
}

int sketch_report_listen(void* arg) {
    const unsigned buf_size = MAX_BATCH * sizeof(struct flow_key);
    struct socket *sock, *client_sock;
    struct sockaddr_in s_addr;
    int ret = 0;

    memset(&s_addr, 0, sizeof(s_addr));
    s_addr.sin_family = AF_INET;
    s_addr.sin_port = htons(listen_port);
    s_addr.sin_addr.s_addr = htonl(INADDR_ANY);


    //sock = (struct socket *)kmalloc(sizeof(struct socket), GFP_KERNEL);
    //client_sock = (struct socket *)kmalloc(sizeof(struct socket), GFP_KERNEL);

    /*create a socket*/
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM, 0, &sock);
    if (ret) {
        printk("server:socket_create error!\n");
    }
    printk("server:socket_create ok!\n");

    /*bind the socket*/
    ret = kernel_bind(sock, (struct sockaddr *)&s_addr, sizeof(struct sockaddr_in));
    if (ret < 0) {
        printk("server: bind error\n");
        return ret;
    }
    printk("server:bind ok!\n");

    /*listen*/
    ret = kernel_listen(sock, 10);
    if (ret < 0) {
        printk("server: listen error\n");
        return ret;
    }
    printk("server:listen ok!\n");
    /*kmalloc a receive buffer*/
    char* recvbuf = NULL;
    recvbuf = kzalloc(buf_size, GFP_KERNEL);
    if (recvbuf == NULL) {
        printk("server: recvbuf kmalloc error!\n");
        return -1;
    }
    memset(recvbuf, 0, sizeof(recvbuf));

    //set_current_state(TASK_INTERRUPTIBLE);
    while (!kthread_should_stop()) {
        ret = kernel_accept(sock, &client_sock, O_NONBLOCK);
        if (ret < 0) {
            if (ret == -EAGAIN) {
                usleep_range(5000, 10000);
                schedule();
                continue;
            }
            else {
                printk("server:accept error!\n");
                return ret;
            }
        }
        printk("server: accept ok, Connection Established\n");


        /*receive message from client*/
        struct kvec vec;
        struct msghdr msg;
        memset(&vec, 0, sizeof(vec));
        memset(&msg, 0, sizeof(msg));
        vec.iov_base = recvbuf;
        vec.iov_len = buf_size;
        ret = kernel_recvmsg(client_sock, &msg, &vec, 1, buf_size, 0); /*receive message*/
        printk("receive message, length: %d\n", ret);
        get_stat_and_send_back(client_sock, recvbuf, ret, (struct sketch_q*)arg);
        /*release socket*/
        sock_release(client_sock);
    }
    sock_release(sock);
    kfree(recvbuf);
    return ret;
    return 0;
}

struct task_struct* listen_thread;
struct sketch_q q;
int sketch_report_init(void* sketch, elemtype(*query)(void* this, struct flow_key* key)) {
    q.instance = sketch;
    q.query = query;
    listen_thread = kthread_create(sketch_report_listen, &q, "listen thread");
    wake_up_process(listen_thread);
    return 0;
}


int sketch_report_clean(void) {
    kthread_stop(listen_thread);
    return 0;
}
