I want u to create an agent crawler to download PDF from https://putusan3.mahkamahagung.go.id, but beware there is cloudflare anti robot so u should use playwright or something
similar. [Image #1] after that in this image https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1.html it should navigate to pick individual case, sometime when
clicking into detail of the case cloudflare is triggered again [Image #2] this is how the detail page looks like and u should look for download pdf to donwload the pdf https://
putusan3.mahkamahagung.go.id/direktori/putusan/zaf14cd81bd6894491e8303832343038.html here is the element if needed <ul class="portfolio-meta nobottommargin">
<li><span><i class="icon-download"></i>Download Zip</span></li>
<li>
</li><li><a href="https://putusan3.mahkamahagung.go.id/direktori/download_file/bbe7ff2bcfe58bdadb9c577d880c027e/zip/zaf14cd81bd6894491e8303832343038">42/Pid.Sus/2026/PN_Pya.zip</a></li>

                                        <li><span><i class="icon-files"></i>Download PDF</span></li>
                                        <li>
                                                                                                </li><li><a href="https://putusan3.mahkamahagung.go.id/direktori/download_file/bbe7ff2bcfe58bdadb9c577d880c027e/pdf/zaf14cd81bd6894491e8303832343038">42/Pid.Sus/2026/PN_Pya.pdf</a></li>

                                    </ul>.use uv to manage all dependencies
