# guild_follow_lock(VAR LOCKED DOC): a backend revision cache variable that follows backends.lock.json.
#
# The revisions used to be plain CACHE variables set from the lock, so an existing build directory kept the revision
# of its first configure after the lock moved on, and built the old sources without a word.  Now VAR is reset to the
# lock's revision whenever it still holds the revision the lock had at the previous configure; a revision passed by
# hand (-DVAR=...) is kept.  A build directory from before this helper (VAR cached, no record of the lock) follows
# the lock once.
function(guild_follow_lock var locked doc)
  set(seen "_${var}_LOCK_SEEN")
  if(NOT DEFINED ${var})
    set(follow TRUE)
  elseif(DEFINED ${seen})
    if("${${var}}" STREQUAL "${${seen}}")
      set(follow TRUE)
    endif()
  elseif(EXISTS "${CMAKE_BINARY_DIR}/CMakeCache.txt" AND NOT "${${var}}" STREQUAL "${locked}")
    message(STATUS "guild: ${var} follows backends.lock.json: ${${var}} -> ${locked}")
    set(follow TRUE)
  endif()
  if(follow)
    set(${var} "${locked}" CACHE STRING "${doc}" FORCE)
  endif()
  set(${seen} "${locked}" CACHE INTERNAL "backends.lock.json's revision at the last configure")
endfunction()
