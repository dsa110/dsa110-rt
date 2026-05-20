/* dsart_capture_mon.c --- POSIX-shm monitoring publisher.
 *
 * Implementation of the dsart_capture_mon.h surface. Public functions
 * are:
 *   - dsart_capture_mon_open(): O_CREAT a shm segment, mmap, zero-init.
 *   - dsart_capture_mon_tick(): stamp last_update_utc_ns (called ~10 Hz
 *     from the stats_thread; the sidecar's watchdog gates on > 1 s
 *     staleness to declare the binary degraded).
 *   - dsart_capture_mon_set_arm(): atomic arm_state transition.
 *
 * The binary writes counters via the standard `atomic_fetch_add` /
 * `atomic_store_explicit` ops directly on the struct fields (see
 * dsart_capture_manythread.c). Only the open/setup path needs a
 * dedicated helper; the rest is straight-line atomic intrinsics that
 * inline at the call site.
 *
 * Cleanup: registered with atexit() so a clean SIGTERM / orchestrator
 * stop unlinks the segment. If the binary segfaults or is SIGKILLed
 * the segment stays in /dev/shm/ and a re-open by the next instance
 * re-uses it (and overwrites the counters with fresh zeroes). The
 * Python sidecar treats a stale shm as 'degraded' via the heartbeat
 * staleness check.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "dsart_capture_mon.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static char _mon_shm_name[64] = "";

static uint64_t _now_utc_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0ULL;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void _unlink_atexit(void)
{
    if (_mon_shm_name[0] != '\0') {
        (void)shm_unlink(_mon_shm_name);
    }
}

dsart_capture_mon_t *dsart_capture_mon_open(int udp_port, int control_port)
{
    char name[64];
    snprintf(name, sizeof(name), DSART_CAPTURE_MON_SHM_FMT, udp_port);

    /* Always start from a fresh segment: unlink any leftover first. */
    (void)shm_unlink(name);

    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0666);
    if (fd < 0) {
        syslog(LOG_WARNING, "dsart_capture_mon: shm_open(%s) failed: %s",
               name, strerror(errno));
        return NULL;
    }

    if (ftruncate(fd, (off_t)sizeof(dsart_capture_mon_t)) != 0) {
        syslog(LOG_WARNING, "dsart_capture_mon: ftruncate(%s) failed: %s",
               name, strerror(errno));
        close(fd);
        (void)shm_unlink(name);
        return NULL;
    }

    void *p = mmap(NULL, sizeof(dsart_capture_mon_t),
                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);  /* mapping survives the close */
    if (p == MAP_FAILED) {
        syslog(LOG_WARNING, "dsart_capture_mon: mmap(%s) failed: %s",
               name, strerror(errno));
        (void)shm_unlink(name);
        return NULL;
    }

    dsart_capture_mon_t *mon = (dsart_capture_mon_t *)p;
    memset(mon, 0, sizeof(*mon));

    atomic_store_explicit(&mon->magic, DSART_CAPTURE_MON_MAGIC,
                          memory_order_release);
    atomic_store_explicit(&mon->version, DSART_CAPTURE_MON_VERSION,
                          memory_order_release);
    atomic_store_explicit(&mon->udp_port, (uint32_t)udp_port,
                          memory_order_release);
    atomic_store_explicit(&mon->control_port, (uint32_t)control_port,
                          memory_order_release);
    atomic_store_explicit(&mon->pid, (uint64_t)getpid(),
                          memory_order_release);
    atomic_store_explicit(&mon->startup_utc_ns, _now_utc_ns(),
                          memory_order_release);
    atomic_store_explicit(&mon->last_update_utc_ns, _now_utc_ns(),
                          memory_order_release);
    atomic_store_explicit(&mon->arm_state, (uint32_t)DSART_ARM_WAITING_FOR_ARM,
                          memory_order_release);

    /* Register the cleanup; safe to do multiple times since we update
     * _mon_shm_name in place (only one capture instance per binary). */
    strncpy(_mon_shm_name, name, sizeof(_mon_shm_name) - 1);
    _mon_shm_name[sizeof(_mon_shm_name) - 1] = '\0';
    static int atexit_registered = 0;
    if (!atexit_registered) {
        atexit(_unlink_atexit);
        atexit_registered = 1;
    }

    syslog(LOG_INFO,
           "dsart_capture_mon: opened shm %s (pid=%lu, version=%u)",
           name, (unsigned long)getpid(), DSART_CAPTURE_MON_VERSION);
    return mon;
}

void dsart_capture_mon_tick(dsart_capture_mon_t *mon)
{
    if (!mon) return;
    atomic_store_explicit(&mon->last_update_utc_ns, _now_utc_ns(),
                          memory_order_release);
}

void dsart_capture_mon_set_arm(dsart_capture_mon_t *mon,
                               dsart_arm_state_t state)
{
    if (!mon) return;
    atomic_store_explicit(&mon->arm_state, (uint32_t)state,
                          memory_order_release);
}
