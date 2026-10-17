#!/usr/bin/env python3
import unittest
from urllib.request import Request
from release_version import ReleaseVersion
from check_guild_update import AssetRedirects,choose


class Updater(unittest.TestCase):
    def test_revision_and_software_order(self):
        manifests={}
        releases=[]
        for tag in ("v0.2.0+r2","v0.2.0+r10","v0.2.1","v0.3.0-rc.1"):
            version=ReleaseVersion.from_tag(tag)
            manifests[tag]={"software_version":version.version,"revision":version.revision,"release_tag":tag}
            releases.append({"tag_name":tag,"html_url":tag,"prerelease":"-rc" in tag,
                             "assets":[{"name":"guild-release.json","browser_download_url":tag}]})
        result=choose(ReleaseVersion("0.2.0",1),releases,"stable",manifests.__getitem__)
        self.assertEqual(result["display_version"],"v0.2.1")
        result=choose(ReleaseVersion("0.2.0",1),releases[:2],"stable",manifests.__getitem__)
        self.assertEqual(result["revision"],10)
        result=choose(ReleaseVersion("0.2.0",1),releases,"preview",manifests.__getitem__)
        self.assertEqual(result["version"],"0.3.0-rc.1")

    def test_manifest_mismatch_and_missing_manifest(self):
        release={"tag_name":"v0.2.0+r2","html_url":"release","assets":[]}
        self.assertFalse(choose(ReleaseVersion("0.1.0"),[release],"stable",lambda _:None)["update_available"])
        release["assets"]=[{"name":"guild-release.json","browser_download_url":"manifest"}]
        with self.assertRaises(ValueError):
            choose(ReleaseVersion("0.1.0"),[release],"stable",lambda _:{"software_version":"0.2.0","revision":1})

    def test_stable_does_not_trust_github_prerelease_flag(self):
        release={"tag_name":"v0.3.0-rc.1","html_url":"release","prerelease":False,
                 "assets":[{"name":"guild-release.json","browser_download_url":"manifest"}]}
        self.assertFalse(choose(ReleaseVersion("0.2.0"),[release],"stable",lambda _:self.fail("preview fetched"))["update_available"])

    def test_redirect_strips_api_credential(self):
        request=Request("https://api.github.com/asset",headers={"Authorization":"Bearer test-only"})
        redirected=AssetRedirects().redirect_request(request,None,302,"Found",{},"https://objects.githubusercontent.com/asset")
        self.assertIsNone(redirected.get_header("Authorization"))


if __name__=="__main__":unittest.main()
