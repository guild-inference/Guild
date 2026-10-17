#!/usr/bin/env python3
"""Check structured Core releases; compare software first, then integer revision.

This reports an available release and never replaces a running executable.
Only releases carrying the structured manifest are eligible; old comparison
artifacts and SDK/kernel integration tags are not interpreted as Core releases.
"""
import argparse
import json
import os
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.parse import urlparse
from release_version import ReleaseVersion


class AssetRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        if urlparse(url).scheme!="https":raise ValueError("release redirects must use HTTPS")
        redirected=super().redirect_request(request,fp,code,message,headers,url)
        if redirected is not None and urlparse(url).hostname!=urlparse(request.full_url).hostname:
            redirected.remove_header("Authorization")
        return redirected


def fetch(url):
    if urlparse(url).scheme!="https":raise ValueError("release downloads must use HTTPS")
    headers={"Accept":"application/vnd.github+json","User-Agent":"Guild-release-check"}
    token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    # Public asset downloads may redirect to storage hosts. Do not forward an
    # API credential to release assets or arbitrary manifest URLs.
    if token and urlparse(url).hostname=="api.github.com":headers["Authorization"]="Bearer "+token
    with build_opener(AssetRedirects()).open(Request(url,headers=headers),timeout=30) as response:
        return json.load(response)


def choose(current,releases,channel,manifest_loader):
    candidates=[]
    for release in releases:
        if release.get("draft") or (channel=="stable" and release.get("prerelease")):
            continue
        try:version=ReleaseVersion.from_tag(release["tag_name"])
        except ValueError:continue
        if channel=="stable" and "-" in version.version:continue
        if version<=current:continue
        assets=[a for a in release.get("assets",[]) if a["name"]=="guild-release.json"]
        if len(assets)!=1:continue
        manifest=manifest_loader(assets[0]["browser_download_url"])
        advertised=ReleaseVersion(manifest["software_version"],manifest["revision"])
        if advertised!=version or manifest.get("release_tag")!=release["tag_name"]:
            raise ValueError("release manifest/tag identity mismatch")
        candidates.append((version,release["html_url"],manifest))
    if not candidates:return {"update_available":False}
    version,url,manifest=max(candidates,key=lambda c:c[0])
    return {"update_available":True,"version":version.version,"revision":version.revision,
            "display_version":version.display,"url":url,"manifest":manifest}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current",required=True,help="immutable version tag, e.g. v0.2.0+r1")
    p.add_argument("--channel",choices=["stable","preview"],default="stable")
    p.add_argument("--releases-json",type=Path,help="offline GitHub release list fixture")
    args=p.parse_args()
    if args.releases_json:
        releases=json.loads(args.releases_json.read_text())
    else:
        releases=[]
        page=1
        while True:
            chunk=fetch(f"https://api.github.com/repos/guild-inference/Guild/releases?per_page=100&page={page}")
            releases.extend(chunk)
            if len(chunk)<100:break
            page+=1
    print(json.dumps(choose(ReleaseVersion.from_tag(args.current),releases,args.channel,fetch),indent=2))


if __name__=="__main__":main()
