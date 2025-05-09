#!/bin/zsh

DBDIR="./.pycor/pgdata"
LOGFILE="$DBDIR/logfile"
PORT=5432
USER="localuser"
DB="localdb"

init() {
  mkdir -p "$DBDIR"
  initdb -D "$DBDIR"
  pg_ctl -D "$DBDIR" -o "-p $PORT" -l "$LOGFILE" start
  sleep 2
  createuser -p $PORT -s "$USER"
  createdb -p $PORT -O "$USER" "$DB"
  pg_ctl -D "$DBDIR" stop
}

start() {
  pg_ctl -D "$DBDIR" -o "-p $PORT" -l "$LOGFILE" start
}

stop() {
  pg_ctl -D "$DBDIR" stop
}

status() {
  pg_ctl -D "$DBDIR" status
}

remove() {
  if pg_ctl -D "$DBDIR" status >& /dev/null; then
    pg_ctl -D "$DBDIR" stop
  fi
  rm -rf "$DBDIR"
  echo "Removed $DBDIR"
}

case "$1" in
  init)   init ;;
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  remove) remove ;;
  *) echo "Usage: $0 {init|start|stop|status|remove}" ;;
esac
